from __future__ import annotations

import abc
import warnings

from typing import TYPE_CHECKING, Union, Dict, Tuple

import arviz as az

import numpy as np
import optax
import pandas as pd
import mudata as md

import jax
import jax.lax
from jax import random, jit, lax
from jax.nn import logsumexp

import jax.numpy as jnp

from optax import exponential_decay, linear_schedule

import numpyro as npy
import numpyro.distributions as npd

import statsmodels.formula.api as smf
from numpyro.infer.svi import SVIRunResult
from sklearn.cluster import KMeans
import tqdm

from dextrademixer.model import ApMHCDeconvolution
from dextrademixer.utils import RegisteredModel

if TYPE_CHECKING:
    from jax._src.typing import Array

npy.enable_x64()

FLOAT_DTYPE = "float64"
INT_DTYPE = "int32"


class DextraDemixer(ApMHCDeconvolution):
    """
    This class implements several mixture models to infer pMHC dextramer specificity from single cell immune profiling
    data with increasing usage of information

    **Given**:

    A read count matrix ***$X_{ij}\in \mathbb{N}$*** approximating the avidity of $i\in N$ T cell for the $j\in M$
    epitope. The $N$ T cells can be grouped based on their T-cell receptor sequence into $C$ cluster.

    **Assumption**:

    1) Each read counts $X_{.j}$ of an epitope is iid.
    2) We assume that $X_{.j}$ can be represented as a mixture of two Negative Binomial distributions. Each clonal group
       $c \in C$ belongs to either the clone-specific **Binding** or the **Non-Binding** component. The **Non-Binding**
       component represents unspecific epitope binding and assay noise.
    3) It is assumed that unspecific binding T cells and non-binding T cells exhibit lower read counts compared to
       specifically binding T cell after appropriate normalization.
    4) T cells of a clonal group $c\in C$ are drawn from the same distribution.

    """

    def __init__(self, model_type: str = "mixturemodel", mode: str = "H"):
        super().__init__()

        if mode.upper() not in ("H", "I", "C"):
            raise ValueError(f"`mode` must be either of the three `I`=independent, `H`=hierarchical, "
                             + f"`C`=clonotype-specific but was {mode}")

        self.sampler = None
        self.trace = None
        self.is_svi = None
        self.svi_result = None
        self.rng_key = None
        self.guide = None
        self.mode = mode.upper()

        if model_type not in ADextraDemixerModel.registry.keys():
            raise warnings.warn(f"`model_type` {model_type} not supported using the standard model.")
        self.model = ADextraDemixerModel.registry.get(model_type, DextraDemixerMixtureModel)()

    @property
    def version(self):
        return self.model.version

    @property
    def model_type(self):
        return self.model.name

    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available DextraDemixer models and their supported versions

        :return: list(str) - list of DextraDemixer models represented as string
        """
        return [k for k in ADextraDemixerModel.registry.keys()]

    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_key: str,
                              gex_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_key: str = None,
                              ir_cov_key: str = None,
                              **kwargs):
        """
        Preprocesses the data and initializes the model

        Args:
            mdata: A Mudata containing only dextramer counts and clonotype information
            pmhc_key: a string specifying the pMHC column in `gex_key` modality`s `X` which should be deconvolved
            gex_key: the MuData transcriptome module key
            neg_ctrl_key: (Optional) a string specifying the negative control column in `gex_key` modality`s `X`
            ir_key: the MuData AIRR module key
            ir_clone_key: (Optional) a string specifying the field in `obs` of `ir_key` that holds clonotype ids
            ir_cov_key: (Optional) the key in AIRR module's `.uns` that contains a full, symmetric and square distance matrix
                         for all clonotype cluster
            kwargs: dictionary of additional information pasted to the Model object (used for custom model prior)
        """
        gex = mdata.mod[gex_key]
        air = mdata.mod[ir_key]
        N = gex.shape[0]

        x = gex[:, pmhc_key].X.toarray().reshape((N,))
        x_neg = gex[:, neg_ctrl_key].X.toarray().reshape((N,)) if neg_ctrl_key else None

        c = air.obs[ir_clone_key].to_numpy().astype("int32") if ir_clone_key is not None else None
        sigma = air.uns[ir_cov_key] if ir_cov_key is not None else None

        if self.mode == "C":
            if c is None:
                raise ValueError("If `mode`= C a clonotype vector `c` must be specified.")

        self._check_parameters(x, x_neg, c, sigma)
        self.model.preprocess_model_data(x, x_neg, c, sigma, self.mode, **kwargs)

    @staticmethod
    def get_default_sampler_config():

        sampler_config = {
            "mcmc": {
                "num_samples": 1000,
                "num_warmup": 1000,
                "num_chains": 4,
                "progress_bar": False,
                "nuts": {
                    "target_accept_prob": 0.9,
                    "max_tree_depth": 15
                }
            },
            "svi": {
                "maxiter": 5000,
                "progress_bar": True,
                "adam": {
                    "init_value": 1e-2,
                    "transition_steps": 1000,
                    "decay_rate": 0.99,
                    "end_value": 1e-7
                },
                "tracer": {
                    "num_particles": 10,
                }
            }
        }

        return sampler_config

    def fit(self, sampler_config: Dict[str, Union[int, float]] = None, rng_key: int = 3) -> az.InferenceData:
        """
        fits the mixture model with MCMC and returns the trace
        """
        if self.model.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        if sampler_config is None:
            sampler_config = self.get_default_sampler_config()["mcmc"]

        nuts_config = {**self.get_default_sampler_config()["mcmc"]["nuts"], **sampler_config.get("nuts", {})}
        sampling_config = {**self.get_default_sampler_config()["mcmc"], **sampler_config}
        sampling_config.pop("nuts", None)

        self.sampler = npy.infer.MCMC(
            npy.infer.NUTS(self.model.model, **nuts_config),
            **sampling_config
        )

        self.sampler.run(random.PRNGKey(rng_key))

        return self.__make_arvis()

    def fit_svi(self, guide=npy.infer.autoguide.AutoNormal, svi_config: Dict[str, Union[int, float]] = None,
                nof_inits: int = 100, use_minimal_loss: bool = True, rng_key: int = 998777,
                return_loss: bool = False) \
            -> az.InferenceData:
        """
        Implements stochastic variational inference

        guide: The guide to use for variational inference. If None, self.model object will be checked for a guide function
        svi_config: configuration for optimizer (Adam) and posterior samples
        nof_inits: number of initializations tried with different seeds to find gut init values
        use_minimal_loss: boolean indicating whether to report the parameters with the lowest loss instead
        rng_key: integer seed to initialize numpyros RNG-Key store
        """

        if self.model.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.is_svi = True
        self.rng_key = rng_key

        if svi_config is None:
            svi_config = self.get_default_sampler_config()["svi"]

        adam_config = {**self.get_default_sampler_config()["svi"]["adam"], **svi_config.get("adam", {})}
        tracer_config = {**self.get_default_sampler_config()["svi"]["tracer"], **svi_config.get("tracer", {})}
        svi_config = {**self.get_default_sampler_config()["svi"], **svi_config}
        svi_config.pop("adam", None)
        svi_config.pop("tracer", None)

        # check for custom guide in self.model otherwise use autoguide
        # optimizer = npy.optim.ClippedAdam(exponential_decay(**adam_config),
        #                                   b1=svi_config['adam_beta']['b1'], b2=svi_config['adam_beta']['b2'])
        optimizer = npy.optim.ClippedAdam(adam_config['init_value'])

        # find good random initialization
        random_init = []
        for i, key in enumerate(random.split(random.PRNGKey(rng_key), nof_inits)):
            if callable(getattr(self.model, "guide", None)):
                self.guide = self.model.guide
            else:
                self.guide = guide(self.model.model, init_loc_fn=npy.infer.initialization.init_to_median)
            svi = npy.infer.SVI(self.model.model, self.guide, optimizer,
                                loss=npy.infer.TraceGraph_ELBO(**tracer_config))
            init_state = svi.init(key)
            loss = svi.evaluate(init_state)

            # Initialization depends on the guide, so need to save the best guide
            random_init.append((loss, key, self.guide))

        init_losses = np.array([x[0] for x in random_init])
        best_idx = jnp.nanargmin(init_losses)
        best_loss, best_key, best_guide = random_init[best_idx]

        self.guide = best_guide
        svi = npy.infer.SVI(self.model.model, self.guide, optimizer,
                            loss=npy.infer.TraceGraph_ELBO(**tracer_config))

        # DEBUG
        # print("best_loss:", best_loss, "best_key:", best_key)

        # Old implementation
        # self.svi_result = svi.run(rng_key=random.PRNGKey(best_key),
        #                           num_steps=svi_config.get("maxiter", 1000),
        #                           stable_update=True,
        #                           progress_bar=svi_config.get("progress_bar", False))

        def body_fn(svi_state, step):
            svi_state, loss = svi.stable_update(svi_state, step=step)
            return svi_state, loss, svi.get_params(svi_state)

        svi_state = svi.init(rng_key=best_key)
        losses = []
        params = []
        with tqdm.trange(1, svi_config.get("maxiter", 1000) + 1,
                         disable=(not svi_config.get("progress_bar", False)), mininterval=10) as t:
            batch = max(svi_config.get("maxiter", 1000) // 20, 1)
            for i in t:
                svi_state, loss, param = jit(body_fn)(svi_state, i)

                losses.append(loss)
                params.append(param)
                if i % batch == 0:
                    valid_losses = [x for x in losses[i - batch:] if x == x]
                    num_valid = len(valid_losses)
                    if num_valid == 0:
                        avg_loss = float("nan")
                    else:
                        avg_loss = sum(valid_losses) / num_valid
                    t.set_postfix_str(
                        "init loss: {:.4f}, avg. loss [{}-{}]: {:.4f}".format(
                            losses[0], i - batch + 1, i, avg_loss
                        ),
                        refresh=False,
                    )
        losses = jnp.stack(losses)
        params = params[jnp.nanargmin(losses)] if use_minimal_loss else params[-1]
        self.svi_result = SVIRunResult(params=params, losses=losses, state=svi_state)

        posterior_samples = self.guide.sample_posterior(random.PRNGKey(self.rng_key), self.svi_result.params,
                                                        sample_shape=(500,))

        # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
        posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
        inference_data = az.from_dict(posterior=posterior_samples_np)

        if return_loss:
            converged = not jnp.isnan(losses).all()
            best_iteration = jnp.nanargmin(losses)
            best_loss = losses[best_iteration]
            return inference_data, {"init_loss": losses[0], "converged": converged,
                                    "best_loss": best_loss, "best_iteration": best_iteration}
        else:
            return inference_data

    def predict_posterior_class(self,
                                threshold: float = None,
                                target_fdr: float = None
                                ) -> Tuple[Array, Array]:
        """
        Returns the binder assignments based on the inferred posterior class probabilities.
        Assignment can be either be done by providing a threshold or target fdr value if FDR control is wanted.
        If neither threshold nor target_fdr is provided the max posterior class probability will be used.

        Args:
             threshold: (Optional) a threshold in [0,1] determining binder based on inferred posterior class
                        probabilities
            target_fdr: (Optional) the FDR threshold to control False discovery rate based on the posterior
                        class probability
        Returns:
            A tuple (p, assignment) of arrays with p being the posterior probability of binding and assignment the
            class assignment decision
        """
        if self.sampler is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        # posterior probability of belonging to the binding class
        if self.is_svi:
            predictive = npy.infer.Predictive(self.model.model, guide=self.guide, params=self.svi_result.params,
                                              num_samples=500)
            samples = predictive(jax.random.PRNGKey(self.rng_key))  # self.rng_key
            p = jnp.exp(jnp.nanmean(samples["log_p"], axis=0))[:, 1]

        else:
            p = jnp.mean(jnp.exp(self.sampler.get_samples()["log_p"][..., 1]), axis=0)

        assignment = self._predict_posterior_class(p, threshold, target_fdr)

        return p, assignment

    def summary(self):
        if self.trace is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

        if self.is_svi:
            posterior_samples = self.guide.sample_posterior(random.PRNGKey(self.rng_key), self.svi_result.params,
                                                            sample_shape=(500,))

            # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
            posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
            inference_data = az.from_dict(posterior=posterior_samples_np)
            return az.summary(inference_data, var_names=["~log_p"])
        else:
            return az.summary(self.trace, var_names=["~log_p"])

    def __make_arvis(self):
        self.trace = az.from_numpyro(self.sampler)
        return self.trace


class ADextraDemixerModel(metaclass=RegisteredModel):
    """
    Abstract model class of DextraDemixer
    """

    def __init__(self):
        self.mode = None
        self._name = "Abstract"
        self._version = "0.0.0"
        self._data = None
        self._kmeans_dict = None

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, Array],
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              **kwargs):
        """
        """

        self.data = {"x": jnp.array(x, dtype=INT_DTYPE),
                     "x_neg": None if neg_cont is None else jnp.array(neg_cont, dtype=FLOAT_DTYPE),
                     "clone": None if c is None else jnp.array(c, dtype=INT_DTYPE),
                     "sigma": None if sigma is None else jnp.array(sigma, dtype=FLOAT_DTYPE),
                     }

        self.mode = mode

    def _init_kmeans(self, scale_factor=1.0) -> Dict:
        """
        Initialize KMeans with 2 clusters and compute all necessary prior parameters
        for mu_q, sigma_q, alpha, and tau priors.

        This method calculates the following priors:
        - mu_q_mean_prior: Mean of each cluster (KMeans cluster centers)
        - mu_q_var_prior: Variance of each cluster (spread of the cluster points)
        - alpha_var_prior: Dispersion (or precision) for each cluster
        - tau_concentration_prior: Proportion of each cluster in the dataset

        returns: Dict with params estimates and  k-mean labels,
        """
        x = self.data["x"]
        clone = self.data.get("clone", None)
        sigma = self.data.get("sigma", None)
        n_clusters = 2  # KMeans with 2 clusters

        # Perform KMeans clustering
        # TODO Workaround if there is an outlier in the data which becomes a sole cluster
        for seed in range(100):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(x.reshape(-1, 1))
            labels = kmeans.labels_
            if (labels == 0).sum() > 1 and (labels == 1).sum() > 1:
                break

        # Initialize lists for cluster attributes
        cluster_means = []
        cluster_variances = []

        kmeans_dict = {}

        # Calculate parameters for each cluster
        for cluster_id in range(n_clusters):
            cluster_points = x[labels == cluster_id]

            # Calculate mean (mu_q_mean_prior)
            cluster_mean = np.mean(cluster_points)
            cluster_means.append(cluster_mean)

            # Calculate variance (mu_q_var_prior), using unbiased variance estimator
            cluster_variance = np.var(cluster_points, ddof=1)
            cluster_variances.append(cluster_variance)

        # Calculate cluster proportions (tau_concentration_prior)
        cluster_counts = np.bincount(labels, minlength=2)
        cluster_proportions = cluster_counts / len(labels)

        # Sort clusters by mean for consistency
        sorted_indices = np.argsort(cluster_means)
        cluster_means = np.array(cluster_means)[sorted_indices]
        cluster_variances = np.array(cluster_variances)[sorted_indices]
        cluster_proportions = np.array(cluster_proportions)[sorted_indices]

        if clone is not None and sigma is not None:
            p1 = cluster_proportions[1]
            log_odds_p1 = np.log(p1 / (1 - p1))

            kmeans_dict.update({"mu_w_mean_prior": log_odds_p1,
                                "mu_w_var_prior": jnp.clip(scale_factor * np.std(cluster_proportions),
                                                           0.1, 10.0)})

        # Set mode-specific parameters (mode "C" for cloneotypes)
        if clone is not None:
            unique_clones = jnp.unique(clone)
            clone_proportions = []

            for unique_clone in unique_clones:
                clone_points = x[clone == unique_clone]
                clone_labels = labels[clone == unique_clone]

                # Clone-specific proportions
                clone_cluster_counts = np.bincount(clone_labels, minlength=2)[sorted_indices]
                clone_proportions.append(clone_cluster_counts / len(clone_points))

            cluster_proportions = np.array(clone_proportions)

        # Update model configuration with calculated priors
        kmeans_dict.update({
            "z": labels,
            "mu_q_mean_prior": np.log(cluster_means),  # Mean for each cluster
            "mu_q_var_prior": np.maximum(np.log(np.sqrt(cluster_variances)), 5.0),  # Standard deviation for each cluster
            "cluster_proportion": cluster_proportions,
            "tau_concentration_prior": cluster_proportions * 10 + 1,  # Concentration for Dirichlet prior
        })

        return kmeans_dict

    @abc.abstractmethod
    def model(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_default_model_config(self) -> Dict:
        return {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        return self._name

    @property
    @abc.abstractmethod
    def version(self) -> str:
        return self._version

    @property
    def data(self) -> Dict:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class DextraDemixerMixtureModel(ADextraDemixerModel):
    """
    Implements a two-component negative-binomial mixture model of the form:

    The generative model is wlog. specified for data of an individual epitope $j$.

    $$\begin{align}
    \text{\# Hyperprior}\\
    &\mu_q \sim \text{N}(0,10)\\
    &\sigma_q \sim \text{HC}(5)\\
    &q^b \sim \text{N}(\mu_q, \sigma_q) &\forall b \in \{0,1\}\\
    \text{IF mode == "H":}\\
    &\alpha \sim \text{HC}(10)\\
    \text{ELSE:}\\
    &\alpha^b \sim \text{HC}(10) &\forall b \in \{0,1\}\\
    \text{IF C defined:}&\\
    \text{IF $\Sigma$ defined:}&\\
    &w \sim \text{MvLogitN}(0, \Sigma)\\
    \text{ELSE:}\\
    &w_c \sim \text{Dir}([1,1]) &\forall c \in C\\
    \text{IF mode == "H":}\\
    &X_{ij}\sim (1-w^0_{C(i)})\text{NegBinom}({s_i}*e^{q^0}, \alpha) + w^1_{C(i)}\text{NegBinom}({s_i}*e^{q^1}, \alpha)&\forall i \in N\\
    \text{ELIF mode == "C":}\\
    &X_{ij}\sim (1-w^0_{C(i)})\text{NegBinom}({s_i}*e^{q^0}, \alpha_{C(i)}) + w^1_{C(i)}\text{NegBinom}({s_i}*e^{q^1}, \alpha_{C(i)})&\forall i \in N\\
    \text{ELSE:}\\
    &X_{ij}\sim (1-w^0_{C(i)})\text{NegBinom}({s_i}*e^{q^0}, \alpha^0) + w^1_{C(i)}\text{NegBinom}({s_i}*e^{q^1}, \alpha^1)&\forall i \in N\\
    \text{ELSE:}\\
    &w \sim \text{Dir}([1,1])\\
    \text{IF mode == "H":}\\
    &X_{ij}\sim (1-w^0)\text{NegBinom}({s_i}*e^{q^0}, \alpha) + w^1\text{NegBinom}({s_i}*e^{q^1}, \alpha)&\forall i \in N\\
    \text{ELSE:}\\
    &X_{ij}\sim (1-w^0)\text{NegBinom}({s_i}*e^{q^0}, \alpha^0) + w^1\text{NegBinom}({s_i}*e^{q^1}, \alpha^1)&\forall i \in N\\
    \text{IF $X_{.j}^{\text{neg}}$ defined:}\\
    \text{IF mode == "H":}\\
    &X_{.j}^{\text{neg}} \sim \text{NegBinom}({s_i}^{\text{neg}}*e^{q^0}, \alpha)&\forall i \in N_{\text{neg}}\\
    \text{ELIF mode == "C":}\\
    &X_{.j}^{\text{neg}} \sim \text{NegBinom}({s_i}^{\text{neg}}*e^{q^0}, \alpha[C_{\text{neg}}(i)])&\forall i \in N_{\text{neg}}\\
    \text{ELSE:}\\
    &X_{.j}^{\text{neg}} \sim \text{NegBinom}({s_i}^{\text{neg}}*e^{q^0}, \alpha^0)&\forall i \in N_{\text{neg}}\\
    \text{s.t.  }  q_{\text{raw}}^0 &< q_{\text{raw}}^1\\
    \end{align}$$

    with $C(i)$ being the clone index of T cell $i$ and $\hat{s}_i$ a cell-specific scaling factor accounting for
    differences in sequencing depth. $q_c$ is the expectation value of avidity for clone $c\in C$ or the expacted
    value of unspecific binding of epitope $i$.
    """

    def __init__(self):
        super().__init__()
        self._name = "mixturemodel"
        self._version = "0.0.3"
        self._model_config = {
            "mu_w_mean_prior": 0.0,
            "mu_w_var_prior": 10.0,
            "mu_q_mean_prior": 0.0,
            "mu_q_var_prior": 5.0,
            "sigma_q_var_prior": 10.0,
            "alpha_var_prior": 10.0,
        }

    def get_default_model_config(self) -> Dict:
        return self._model_config

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def model(  # type: ignore
            self,
            **kwargs):

        if self.data is None:
            raise RuntimeError("Model was not properly initialized. Please call `preprocess_model_data` first")

        model_config = {**self.get_default_model_config(), **kwargs["model_config"]} if (
                "model_config" in kwargs) else self.get_default_model_config()

        x = self.data["x"]
        x_neg = self.data["x_neg"]
        clone = self.data["clone"]
        sigma = self.data["sigma"]

        # plates
        N_sample = x.shape[0]
        c_nof = np.unique(clone).size if clone is not None else 0
        K = 2

        # hyperprior parameters
        mu_w_mean_prior = model_config.get("mu_w_mean_prior", 0.0)
        mu_w_var_prior = model_config.get("mu_w_var_prior", 10.0)
        mu_q_mean_prior = model_config.get("mu_q_mean_prior", 0.0)
        mu_q_var_prior = model_config.get("mu_q_var_prior", 5.0)
        sigma_q_var_prior = model_config.get("sigma_q_var_prior", 10.0)
        alpha_var_prior = model_config.get("alpha_var_prior", 10.0)

        # hyperprior
        mu_q = npy.sample("mu_q", npd.Normal(mu_q_mean_prior, mu_q_var_prior))
        sigma_q = npy.sample("sigma_q", npd.HalfCauchy(sigma_q_var_prior))

        #shape prior
        if self.mode == "H":
            alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                    npd.transforms.PowerTransform(-2.0)))
        elif self.mode == "C":
            with npy.plate("clone_axis", c_nof):
                alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                        npd.transforms.PowerTransform(-2.0)))
        else:
            with npy.plate("cluster_axis", K):
                alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                        npd.transforms.PowerTransform(-2.0)))

        # cluster probability prior
        if clone is not None:
            if sigma is not None:
                # non-centered multivariat parametrization
                mu_w = npy.sample("mu_w", npd.Normal(mu_w_mean_prior, mu_w_var_prior))
                with npy.plate("clone_axis", c_nof):
                    gamma_w = npy.sample("gamma_w", npd.Normal(loc=0, scale=1))
                L = jnp.linalg.cholesky(sigma)
                w_raw = jax.scipy.special.ndtr(jnp.clip(mu_w + jnp.dot(L, gamma_w), -5, 5))
                #w_raw = jax.scipy.special.expit(mu_w + jnp.dot(L, gamma_w))
                w = npy.deterministic("w", jnp.stack([1 - w_raw, w_raw], axis=-1))
            else:
                with npy.plate("clone_axis", c_nof):
                    w = npy.sample("w", npd.Dirichlet(jnp.ones(K)))
            z = npd.Categorical(probs=w[clone])
        else:
            w = npy.sample("w", npd.Dirichlet(jnp.ones(K)))
            z = npd.Categorical(probs=w)

        q = npy.sample("q",
                       npd.TransformedDistribution(npd.LogNormal(loc=mu_q, scale=sigma_q).expand((K,)),
                                                   npd.transforms.OrderedTransform()))

        with npy.plate("sample_axis", N_sample):
            if x_neg is not None:
                if self.mode == "H":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0], concentration=alpha),
                                          obs=x_neg)
                elif self.mode == "C":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0],
                                                                concentration=alpha[clone]),
                                          obs=x_neg)
                else:
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0], concentration=alpha[0]),
                                          obs=x_neg)

            # target pMHC
            if self.mode == "C":
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=q,
                                                                         concentration=alpha[clone].reshape(
                                                                             (N_sample, 1))
                                                                         )
                                                )
            else:
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=q,
                                                                         concentration=alpha))

            yhat = npy.sample("yhat", mixture, obs=x)

            # Until here, where we can track the membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))


class DextraDemixerKmeansModel(ADextraDemixerModel):
    """
    Dextramixer version who is initialized and prior parametrized by K-means++ results
    Thus model does not rely on hyperpriors and is a bit simpler
    """

    def __init__(self):
        super().__init__()
        self._name = "mixturemodelkmeans"
        self._version = "0.0.1"
        self._model_config = {
            "mu_w_mean_prior": 0.0,
            "mu_w_var_prior": 10.0,
            "mu_q_mean_prior": 0.0,
            "mu_q_var_prior": 10.0,
            "sigma_q_var_prior": 10.0,
            "alpha_var_prior": 10.0,
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, Array],
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              scale_factor: float = 1.0,
                              **kwargs):

        super().preprocess_model_data(x, neg_cont, c, sigma, mode, **kwargs)
        self._kmeans_dict = self._init_kmeans(scale_factor=scale_factor)
        self._model_config.update(self._kmeans_dict)

    def get_default_model_config(self) -> Dict:
        return self._model_config

    def model(self, **kwargs):
        """
        Define the probabilistic model based on the preprocessed data and KMeans initialization.
        """
        if self.data is None:
            raise RuntimeError("Model was not properly initialized. Please call `preprocess_model_data` first.")

        model_config = {**self.get_default_model_config(), **kwargs.get("model_config", {})}

        x = self.data["x"]
        x_neg = self.data["x_neg"]
        clone = self.data["clone"]
        sigma = self.data["sigma"]
        N_sample = x.shape[0]
        c_nof = np.unique(clone).size if clone is not None else 0
        K = 2

        # Extract hyperpriors
        mu_w_mean_prior = model_config["mu_w_mean_prior"]
        mu_w_var_prior = model_config["mu_w_var_prior"]
        mu_q_mean_prior = model_config["mu_q_mean_prior"]
        mu_q_var_prior = model_config["mu_q_var_prior"]
        alpha_var_prior = model_config["alpha_var_prior"]
        tau_concentration_prior = model_config["tau_concentration_prior"]

        if self.mode == "H":
            alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                    npd.transforms.PowerTransform(-2.0)))
        elif self.mode == "C":
            with npy.plate("clone_axis", c_nof):
                alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                    npd.transforms.PowerTransform(-2.0)))
        else:
            with npy.plate("cluster_axis", K):
                alpha = npy.sample("alpha", npd.TransformedDistribution(npd.HalfCauchy(alpha_var_prior),
                                                                        npd.transforms.PowerTransform(-2.0)))

        # Cluster probability prior
        if clone is not None:
            if sigma is not None:
                mu_w = npy.sample("mu_w", npd.Normal(mu_w_mean_prior, mu_w_var_prior))
                with npy.plate("clone_axis", c_nof):
                    gamma_w = npy.sample("gamma_w", npd.Normal(loc=0, scale=1))
                L = jnp.linalg.cholesky(sigma)
                w_raw = jax.scipy.special.ndtr(jnp.clip(mu_w + jnp.dot(L, gamma_w), -5, 5))
                w = npy.deterministic("w", jnp.stack([1 - w_raw, w_raw], axis=-1))
            else:
                with npy.plate("clone_axis", c_nof):
                    w = npy.sample("w", npd.Dirichlet(tau_concentration_prior))
            z = npd.Categorical(probs=w[clone])
        else:
            w = npy.sample("w", npd.Dirichlet(tau_concentration_prior))
            z = npd.Categorical(probs=w)

        # kmeans q prior already ordered and OrderedTransform exponentiates the values leading to overflow
        q = npy.sample("q", npd.LogNormal(loc=mu_q_mean_prior, scale=mu_q_var_prior))

        # Sample from the mixture model
        with npy.plate("sample_axis", N_sample):
            if x_neg is not None:
                if self.mode == "H":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0], concentration=alpha),
                                          obs=x_neg)
                elif self.mode == "C":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0],
                                                                concentration=alpha[clone]),
                                          obs=x_neg)
                else:
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=q[0], concentration=alpha[0]),
                                          obs=x_neg)

            # target pMHC
            if self.mode == "C":
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=q,
                                                                         concentration=alpha[clone].reshape(
                                                                             (N_sample, 1))
                                                                         )
                                                )
            else:
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=q,
                                                                         concentration=alpha))

            yhat = npy.sample("yhat", mixture, obs=x)

            # Membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))