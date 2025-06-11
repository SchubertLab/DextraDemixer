from __future__ import annotations

import abc
import warnings
import os

from typing import TYPE_CHECKING, Union, Dict, Tuple

import arviz as az

import numpy as np
import pandas as pd
import mudata as md
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.lax
from jax import random, jit
from jax.nn import logsumexp
import jax.numpy as jnp

import numpyro as npy
import numpyro.distributions as npd

from numpyro.infer.svi import SVIRunResult
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
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

    def __init__(self, model_type: str = "mixturemodel", mode: str = "H", alpha_model="overdispersion"):
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
        self.alpha_model = alpha_model

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
        self.model.preprocess_model_data(x=x, neg_cont=x_neg, c=c, sigma=sigma, mode=self.mode,
                                         alpha_model=self.alpha_model, **kwargs)

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
        #                                   b1=svi_config["adam_beta"]["b1"], b2=svi_config["adam_beta"]["b2"])
        optimizer = npy.optim.ClippedAdam(adam_config["init_value"])

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

    def plot_results(self, assignment, p_pred, y_true, seed, config):
        plt.figure(figsize=(16, 12))

        # Plot data colored in predicted class assignment
        plt.subplot(3, 4, 1)
        sns.histplot(x=self.model.data["x"], hue=assignment,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.title("Predicted class assignment")

        plt.subplot(3, 4, 5)
        sns.histplot(x=self.model.data["x"], hue=assignment,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.yscale("log")
        plt.title("Predicted class assignment log-scale")

        plt.subplot(3, 4, 9)
        sns.scatterplot(x=self.model.data["x"], y=p_pred, hue=assignment,
                        markers={0: ".", 1: "X"})
        sns.despine()
        plt.ylabel("Posterior probability")
        plt.title("Predicted probability and label of UMI count")

        # Plot data colored in true class assignment
        plt.subplot(3, 4, 2)
        sns.histplot(x=self.model.data["x"], hue=y_true,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.title("True class assignment")

        plt.subplot(3, 4, 6)
        sns.histplot(x=self.model.data["x"], hue=y_true,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.yscale("log")
        plt.title("True class assignment log-scale")

        plt.subplot(3, 4, 10)
        sns.scatterplot(x=self.model.data["x"], y=p_pred, hue=y_true,
                        style=y_true, markers={False: ".", True: "X"})
        sns.despine()
        plt.ylabel("Posterior probability")
        plt.title("Predicted probability of UMI count with true label")

        # Plot posterior distribution of Negative Binomial
        predictive = npy.infer.Predictive(self.guide, params=self.svi_result.params, num_samples=1000)
        posterior_samples = predictive(jax.random.PRNGKey(seed), data=None)

        # Extract mean from posterior samples
        q = posterior_samples["q"].mean(0)
        if self.alpha_model == 'overdispersion':
            overdispersion = posterior_samples["overdispersion"].mean(0)
            alpha = q**2 / (q * (overdispersion + 1) - q)
        else:
            alpha = posterior_samples["alpha"].mean(0)
        w = posterior_samples["w"].mean(0)

        # If mode is C, we average over the clonotypes
        if self.mode == "C":
            # TODO visualize distribution for each clonotype
            alpha = alpha.mean(0)

        x = np.arange(0, self.model.data["x"].max())
        if self.mode == "H" or self.mode == "C":
            alpha0 = alpha
            alpha1 = alpha
        elif self.mode == "I":
            alpha0 = alpha[0]
            alpha1 = alpha[1]

        prob0 = jnp.exp(npd.NegativeBinomial2(q[0], alpha0).log_prob(x))
        prob1 = jnp.exp(npd.NegativeBinomial2(q[1], alpha1).log_prob(x))

        if self.model.data["clone"] is not None:
            # w.shape = (#clonotypes, 2)
            # probX = (max UMI)
            prob0_mix = prob0[None,] * w[:, 0:1]
            prob1_mix = prob1[None,] * w[:, 1:2]
            w_mean = w.mean(0)
            x = np.tile(np.arange(0, self.model.data["x"].max()), (prob0_mix.shape[0]))
        else:
            # w.shape = (2,)
            prob0_mix = prob0 * w[0]
            prob1_mix = prob1 * w[1]
            w_mean = w
        # Mixture of Negative Binomial
        plt.subplot(3, 4, 4)
        sns.lineplot(x=x, y=prob0_mix.reshape(-1),
                     label=f"q={q[0]:.2f} alpha={alpha0:.2f}", linewidth=3)
        sns.lineplot(x=x, y=prob1_mix.reshape(-1),
                    label=f"q={q[1]:.2f} alpha={alpha1:.2f}", linewidth=3)
        sns.lineplot(x=x, y=(prob0_mix + prob1_mix).reshape(-1), linewidth=3, color="k",
                     label=f"mixture w={w_mean[0]:.4f}, {w_mean[1]:.4f}", linestyle="--")
        sns.despine()
        plt.title("Posterior Mixture NB")
        plt.ylabel("Probability")

        # Individual Negative Binomial
        plt.subplot(3, 4, 8)
        sns.lineplot(x=np.arange(0, self.model.data["x"].max()), y=prob0, label=f"q={q[0]:.2f} alpha={alpha0:.2f}")
        sns.lineplot(x=np.arange(0, self.model.data["x"].max()), y=prob1, label=f"q={q[1]:.2f} alpha={alpha1:.2f}")
        sns.despine()
        plt.title("Posterior NB without mixing weights")
        plt.ylabel("Probability")

        if self.model_type == "mixturemodelkmeans":
            cluster_means = self.model._kmeans_dict["cluster_means"]
            cluster_vars = self.model._kmeans_dict["cluster_variances"]
            dists = jnp.abs(self.model.data["x"][:, None].repeat(2, axis=1) - cluster_means)
            labels = np.argmin(dists, axis=1)

            plt.subplot(3, 4, 3)
            sns.histplot(x=self.model.data["x"], hue=labels, discrete=True, element="step", alpha=0.7, stat='percent')
            plt.axvline(cluster_means[0], color="red", linestyle="--")
            plt.axvline(cluster_means[1], color="red", linestyle="--")
            sns.despine()
            plt.title("KMeans clustering")
            plt.ylabel("Percent")

            plt.subplot(3, 4, 7)
            sns.histplot(x=self.model.data["x"], hue=labels, discrete=True, element="step", alpha=0.7, stat='percent')
            plt.yscale("log")
            plt.axvline(cluster_means[0], color="red", linestyle="--")
            plt.axvline(cluster_means[1], color="red", linestyle="--")
            sns.despine()
            plt.title("KMeans cluster log-scale")
            plt.ylabel("Percent")

            plt.subplot(3, 4, 12)
            alpha = cluster_means**2 / (cluster_vars - cluster_means)
            alpha[alpha < 0] = 100
            x = np.arange(0, self.model.data["x"].max())
            prob0 = jnp.exp(npd.NegativeBinomial2(cluster_means[0], alpha[0]).log_prob(x))
            prob1 = jnp.exp(npd.NegativeBinomial2(cluster_means[1], alpha[1]).log_prob(x))

            sns.lineplot(x=x, y=prob0, label=f"q={cluster_means[0]:.2f} alpha={alpha[0]:.2f}")
            sns.lineplot(x=x, y=prob1, label=f"q={cluster_means[1]:.2f} alpha={alpha[1]:.2f}")
            sns.despine()
            plt.title("kMeans determined distribution")
            plt.ylabel("Probability")

        # Save plot
        f1 = f1_score(assignment, y_true)
        plt.suptitle(config.replace("_", " ").replace("ncell", "\nncell") + f"\nF1-score {f1:.3f}")
        os.makedirs("figs", exist_ok=True)
        plt.savefig(f"figs/{config}.png")
        plt.show()
        plt.close()


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
        self.alpha_model = None

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, Array],
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              alpha_model="overdispersion",
                              **kwargs):
        """
        """

        self.data = {"x": jnp.array(x, dtype=INT_DTYPE),
                     "x_neg": None if neg_cont is None else jnp.array(neg_cont, dtype=FLOAT_DTYPE),
                     "clone": None if c is None else jnp.array(c, dtype=INT_DTYPE),
                     "sigma": None if sigma is None else jnp.array(sigma, dtype=FLOAT_DTYPE),
                     }

        self.mode = mode
        self.alpha_model = alpha_model

    def _init_kmeans(self, scale_factor=1.0) -> Dict:
        """
        Initialize KMeans with 2 clusters and compute all necessary prior parameters
        for mu_q, sigma_q, alpha, and tau priors.

        This method calculates the following priors:
        - cluster_means: Mean of each cluster (KMeans cluster centers)
        - cluster_variances: Variance of each cluster (spread of the cluster points)
        - tau_concentration_prior: Proportion of each cluster in the dataset

        returns: Dict with params estimates and  k-mean labels,
        """
        x = self.data["x"].copy()
        # remove outliers
        zscore = (x - x.mean()) / x.std()
        x = x[zscore < 20]  # TODO determine which threshold works the best
        clone = self.data.get("clone", None)
        sigma = self.data.get("sigma", None)
        n_clusters = 2  # KMeans with 2 clusters

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, init=np.vstack([np.min(x), np.max(x)]), n_init="auto").fit(x.reshape(-1, 1))
        labels = kmeans.labels_

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
            "cluster_means": cluster_means,  # Mean for each cluster
            "cluster_variances": cluster_variances,  # variance for each cluster
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
            "var_hyperprior": 10.0,
            "overdispersion_scale_prior": 1.0,
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
                              alpha_model="overdispersion",
                              mode: str = "H",
                              scale_factor: float = 1.0,
                              **kwargs):

        super().preprocess_model_data(x=x, neg_cont=neg_cont, c=c, sigma=sigma, mode=mode,
                                      alpha_model=alpha_model, **kwargs)
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
        cluster_means = model_config["cluster_means"]
        cluster_variances = model_config["cluster_variances"]
        var_hyperprior = model_config["var_hyperprior"]  # Variance for priors if using alpha_model = kmeans
        tau_concentration_prior = model_config["tau_concentration_prior"]
        overdispersion_scale_prior = model_config["overdispersion_scale_prior"]

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

        # Variant 1: Model alpha as through overdispersion
        if self.alpha_model == "overdispersion":
            # Convert kmeans priors to deltas, due to cumsum ordering
            mean_deltas = jnp.array([cluster_means[0], cluster_means[1] - cluster_means[0]])
            var_deltas = jnp.array([cluster_variances[0], max(cluster_variances[1] - cluster_variances[0], 1)])

            # Convert kmeans parameters to lognormal parameters with target mean and variance
            # NB mean parameter: q_prior ~ LogNormal(mu_q, sigma_q), with cluster means and variances
            sigma2_q_prior = jnp.log(var_deltas / mean_deltas ** 2 + 1)
            sigma_q_prior = jnp.sqrt(sigma2_q_prior)
            mu_q_prior = jnp.log(mean_deltas) - sigma2_q_prior / 2

            # Sample delta_q from lognormal distribution and cumsum to create ordered q
            delta_q = npy.sample("q", npd.LogNormal(loc=mu_q_prior, scale=sigma_q_prior))
            q = jnp.cumsum(delta_q, axis=0)

            # NB concentration parameter: alpha = q^2 / (q * overdispersion - q), overdispersion ~ HalfCauchy(1) + 1
            overdispersion_prior_dist = npd.HalfCauchy(overdispersion_scale_prior)

            if self.mode == "C":
                # TODO Doesn't work for C yet, since we somehow would need to have access to the mean of each clone
                with npy.plate("clone_axis", c_nof):
                    overdispersion = npy.sample("overdispersion", overdispersion_prior_dist) + 1
                alpha = npy.deterministic("alpha", q.mean() ** 2 / (q.mean() * overdispersion - q.mean()))
            else:
                with npy.plate("cluster_axis", K):
                    overdispersion = npy.sample("overdispersion", overdispersion_prior_dist) + 1
                alpha = npy.deterministic("alpha", q**2 / (q * overdispersion - q))

        elif self.alpha_model == "kmeans":
            # Variance should roughly follow my guessed "uncertainty" around my prior point in percent,
            # and put adjust the standard deviation at those point sigma^2=(log(1+p))**2
            # e.g., if I think x=30, and I think it should be around 50% of it [15, 45], then I would have
            # sigma^2=log(1+0.5)**2=0.164

            # NB mean parameter: q_prior ~ LogNormal(mu_q, sigma_q), with cluster means and hyperprior variances
            var_hyperprior_deltas = jnp.array([var_hyperprior, 1])  # 1 in delta to have the similar variance for both
            mean_deltas = jnp.array([cluster_means[0], cluster_means[1] - cluster_means[0]])

            sigma2_q_hyperprior = jnp.log(var_hyperprior_deltas / mean_deltas ** 2 + 1)
            sigma_q_hyperprior = jnp.sqrt(sigma2_q_hyperprior)
            mu_q_prior = jnp.log(mean_deltas) - sigma2_q_hyperprior / 2

            delta_q = npy.sample("q", npd.LogNormal(loc=mu_q_prior, scale=sigma_q_hyperprior))
            q = jnp.cumsum(delta_q, axis=0)

            # NB concentration parameter: convert to alpha, alpha_prior ~ LogNormal(mu_alpha, sigma_alpha), with cluster variance and hyperprior variances
            # Calculate alpha parameter with target variance
            alpha_prior = cluster_means**2 / (cluster_variances - cluster_means)
            # In case of underdispersion (negative alpha), set to a high number
            alpha_prior[alpha_prior <= 0] = 100

            var_hyperprior_deltas = jnp.array([var_hyperprior, 1])  # 1 in delta to have similar variance for both
            # Due to cumsum ordering, second component cannot be smaller than first
            alpha_deltas = jnp.array([alpha_prior[0], jnp.maximum(alpha_prior[1] - alpha_prior[0], 1e-5)])

            sigma2_alpha_hyperprior = jnp.log(var_hyperprior_deltas / alpha_deltas ** 2 + 1)
            sigma_alpha_hyperprior = jnp.sqrt(sigma2_alpha_hyperprior)

            mu_alpha_prior = jnp.log(alpha_deltas) - sigma2_alpha_hyperprior / 2

            if self.mode == "C":
                with npy.plate("clone_axis", c_nof):
                    alpha = npy.sample("alpha", npd.LogNormal(loc=mu_alpha_prior.mean(), scale=sigma_alpha_hyperprior[0]))
            else:
                with npy.plate("cluster_axis", K):
                    alpha = npy.sample("alpha", npd.LogNormal(loc=mu_alpha_prior, scale=sigma_alpha_hyperprior))

        else:
            raise NotImplementedError(f" {self.alpha_model} not implemented")

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
