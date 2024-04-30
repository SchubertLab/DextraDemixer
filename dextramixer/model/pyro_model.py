from __future__ import annotations

import abc
import warnings

from typing import TYPE_CHECKING, Literal, Optional, Union, Dict, Tuple

import arviz as az
import jax.lax
import numpy as np
import pandas as pd

import scanpy as sc
import scirpy as ir
import mudata as md

import jax
from jax import random
from jax.nn import logsumexp

import jax.numpy as jnp

import numpyro as npy
import numpyro.distributions as npd
from numpyro.infer import HMC, MCMC, NUTS, initialization

from dextramixer.utils import RegisteredModel

if TYPE_CHECKING:
    from jax._src.prng import PRNGKeyArray
    from jax._src.typing import Array

jax.config.update("jax_enable_x64", True)


class DextraMixer:
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
        if mode.upper() not in ("H", "I", "C"):
            raise ValueError(f"`mode` must be either of the three `I`=independent, `H`=hierarchical, "
                             + f"`C`=clonotype-specific but was {mode}")

        self.sampler = None
        self.trace = None
        self.mode = mode.upper()

        if model_type not in ADextraMixerModel.registry.keys():
            raise warnings.warn(f"`model_type` {model_type} not supported using the standard model.")
        self.model = ADextraMixerModel.registry.get(model_type, DextraMixerMixtureModel)()

    @property
    def version(self):
        return self.model.version

    @property
    def model_type(self):
        return self.model.name

    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available DextraMixer models and their supported versions

        :return: list(str) - list of DextraMixer models represented as string
        """
        return [k for k in ADextraMixerModel.registry.keys()]

    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_key: str,
                              gex_key: str = "gex",
                              neg_cont_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_id: str = None,
                              ir_cov_key: str = None,
                              **kwargs):
        """
        Preprocesses the data and initializes the model

        Args:
            mdata: A Mudata containing only dextramer counts and clonotype information
            pmhc_key: a string specifying the pMHC column in `gex_key` modality`s `X` which should be deconvolved
            gex_key: the MuData transcriptome module key
            neg_cont_key: (Optional) a string specifying the negative control column in `gex_key` modality`s `X`
            ir_key: the MuData AIRR module key
            ir_clone_id: (Optional) a string specifying the field in `obs` of `ir_key` that holds clonotype ids
            ir_cov_key: (Optional) the key in AIRR module's `.uns` that contains a full, symmetric and square distance matrix
                         for all clonotype cluster
            kwargs: dicitionary of additional information pasted to the Model object (used for custom model prior)
        """
        agex = mdata.mod[gex_key]
        air = mdata.mod[ir_key]

        x = agex[:, pmhc_key].X
        x_neg = agex[:, neg_cont_key].X if neg_cont_key else None
        _, size_factor = sc.pp.normalize_total(mdata["gex"], inplace=False).values()
        size_factor = size_factor.reshape(x.shape[0], 1)
        c = air.obs[ir_clone_id].to_numpy().astype("int32") if ir_clone_id is not None else None
        sigma = air.uns[ir_cov_key] if ir_cov_key is not None else None

        self.__check_parameters(x, x_neg, size_factor, c, sigma)
        self.model.preprocess_model_data(x, size_factor, x_neg, c, sigma, self.mode, **kwargs)

    @staticmethod
    def get_default_sampler_config() -> Dict[str, Union[int, float]]:
        """
        """
        sampler_config = {
            "num_samples": 1000,
            "num_warmup": 1000,
            "num_chains": 4,
            "progress_bar": False,
            "nuts": {
                "target_accept_prob": 0.9,
                "max_tree_depth": 15
            }
        }
        return sampler_config

    def fit(self, sampler_config: Dict[str, Union[int, float]] = None, rng: int = 3) -> az.InferenceData:
        """
        fits the mixture model with MCMC and returns the trace
        """
        if self.model is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        if sampler_config is None:
            sampler_config = self.get_default_sampler_config()

        nuts_config = {**self.get_default_sampler_config()["nuts"], **sampler_config.get("nuts", {})}
        sampling_config = {**self.get_default_sampler_config(), **sampler_config}
        sampling_config.pop("nuts", None)

        self.sampler = npy.infer.MCMC(
            npy.infer.NUTS(self.model.model, **nuts_config),
            **sampling_config
        )

        self.sampler.run(random.PRNGKey(rng))

        return self.__make_arvis()

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

        if threshold is not None and target_fdr is not None:
            raise ValueError("Please specify either a manual `threshold` or a `target_fdr` but not both.")

        if threshold is not None and not (0 <= threshold <= 1):
            raise ValueError(f"`threshold`must be in [0,1] but was {threshold}")

        if target_fdr is not None and not (0 <= target_fdr <= 1):
            raise ValueError(f"`target_fdr`must be in [0,1] but was {target_fdr}")

        # posterior probability of belonging to the binding class
        p = jnp.mean(jnp.exp(self.sampler.get_samples()["log_p"][..., 1]), axis=0)

        if target_fdr is not None:
            # Direct posterior probability approach cf. Newton et al.(2004)
            def opt_thresh(p_, alpha):

                incs = p_.copy()
                incs = incs[::-1].sort()

                for c in jnp.unique(incs):
                    fdr = jnp.mean(1 - incs[incs >= c])
                    if fdr < alpha:
                        return c, fdr
                return 1., 0

            threshold, fdr_ = opt_thresh(p, target_fdr)
        if target_fdr is not None or threshold is not None:
            assignment = (p >= threshold).astype("int32")
        else:
            p_ = jnp.mean(jnp.exp(self.sampler.get_samples()["log_p"][...,]), axis=0)
            assignment = jnp.argmax(p_, axis=1)
        return p, assignment

    def __make_arvis(self):
        self.trace = az.from_numpyro(self.sampler)
        return self.trace

    def __check_parameters(self, x, neg_x, size_factor, c, sigma):
        """
        checks consistency of input data before initializing the model
        """

        N = x.shape[0]

        if self.mode == "C":
            if c is None:
                raise ValueError("If `mode`= C a clonotype vector `c` must be specified.")

        if size_factor.shape[0] != N:
            raise ValueError(
                f"`size_factor` and count data `x` require the same size but got {size_factor.shape[0]} and {N}")

        if c is not None:
            if c.shape[0] != N:
                raise ValueError(f"`c` and count data `x` require the same size but got {c.shape[0]} and {N}")

        if neg_x is not None:
            N_neg = neg_x.shape[0]

            if N_neg != N:
                raise ValueError(f"x_neg must have the same size than x but got {N_neg} vs {N}.")

        if sigma is not None:
            if c is None:
                raise ValueError("If `sigma` is given, clonality vector `c` must be given as well")
            else:
                C_nof = len(np.unique(c))
                if sigma.shape[0] != C_nof:
                    raise ValueError(f"Sigma must have shape ({C_nof},{C_nof}) and defined over clonotypes but has"
                                     + f"{sigma.shape}")


class ADextraMixerModel(metaclass=RegisteredModel):
    """
    Abstract model class of DextraMixer
    """

    def __init__(self):
        self.mode = None
        self._name = "Abstract"
        self._version = "0.0.0"
        self._data = None
        self._coords = None

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, Array],
                              size_factor: Union[pd.Series, np.ndarray, Array],
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              **kwargs):
        """
        """
        float_dtype = "float64"
        int_dtype = "int32"

        N = x.shape[0]
        K = 2

        self.data = {"x": jnp.array(x, dtype=int_dtype),
                     "x_neg": None if neg_cont is None else jnp.array(neg_cont, dtype=float_dtype),
                     "size_factor": jnp.array(size_factor, dtype=float_dtype),
                     "clone": None if c is None else jnp.array(c, dtype=int_dtype),
                     "sigma": None if sigma is None else jnp.array(sigma, dtype=float_dtype),
                     }

        self.mode = mode

        # set coord axis
        self.coords = {
            "sample_axis": npy.plate("sample_axis", N, dim=-1),
            "cluster_axis": npy.plate("cluster_axis", K, dim=-1),
        }

        if self.data["clone"] is not None:
            C_nof = len(jnp.unique(self.data["clone"]))
            self.coords["clone_axis"] = npy.plate("clone_axis", C_nof, dim=-1)

        if self.data["x_neg"] is not None:
            N_neg = neg_cont.shape[0]
            self.coords["neg_sample_axis"] = npy.plate("neg_sample_axis", N_neg, dim=-1)

    @abc.abstractmethod
    def model(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_default_model_config() -> Dict:
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
    def coords(self) -> Dict:
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = value

    @property
    def data(self) -> Dict:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class DextraMixerMixtureModel(ADextraMixerModel):
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
        self._version = "0.0.1"

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config = {
            "mu_q_mean_prior": 0.0,
            "mu_q_var_prior": 10.0,
            "sigma_q_var_prior": 10.0,
            "alpha_var_prior": 10.0,
        }
        return model_config

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def model(  # type: ignore
            self,
            **kwargs):

        if self.coords is None:
            raise RuntimeError("Model was not properly initialized. Please call `build_model` first")

        model_config = {**DextraMixerMixtureModel.get_default_model_config(), **kwargs["model_config"]} if (
                "model_config" in kwargs) else DextraMixerMixtureModel.get_default_model_config()

        # data
        x = self.data["x"]
        x_neg = self.data["x_neg"]
        size_factor = self.data["size_factor"].reshape(self.coords["sample_axis"].size, 1)
        clone = self.data["clone"]
        sigma = self.data["sigma"]

        # plates
        sample_axis = self.coords["sample_axis"]
        cluster_axis = self.coords["cluster_axis"]
        K = cluster_axis.size
        if clone is not None:
            clone_axis = self.coords["clone_axis"]
        if x_neg is not None:
            neg_sample_axis = self.coords["neg_sample_axis"]

        # hyperprior parameters
        mu_q_mean_prior = model_config.get("mu_q_mean_prior", 0.0)
        mu_q_var_prior = model_config.get("mu_q_var_prior", 10.0)
        sigma_q_var_prior = model_config.get("sigma_q_var_prior", 10.0)
        alpha_var_prior = model_config.get("alpha_var_prior", 10.0)

        # hyperprior
        mu_q = npy.sample("mu_q", npd.Normal(mu_q_mean_prior, mu_q_var_prior))
        sigma_q = npy.sample("sigma_q", npd.HalfCauchy(sigma_q_var_prior))

        # shape prior
        if self.mode == "H":
            alpha = npy.sample("alpha", npd.HalfCauchy(alpha_var_prior))
        elif self.mode == "C":
            with clone_axis:
                alpha = npy.sample("alpha", npd.HalfCauchy(alpha_var_prior))
        else:
            with cluster_axis:
                alpha = npy.sample("alpha", npd.HalfCauchy(alpha_var_prior))
        npy.factor("power_law", 1 / jnp.sqrt(alpha))  # according to stan standard prior

        # cluster probability prior
        if clone is not None:
            if sigma is not None:
                w_raw = npy.sample("w_raw", npd.TransformedDistribution(
                    npd.MultivariateNormal(covariance_matrix=sigma),
                    npd.transforms.SigmoidTransform()))
                w = npy.deterministic("w", jnp.stack([1 - w_raw, w_raw], axis=-1))
            else:
                with clone_axis:
                    w = npy.sample("w", npd.Dirichlet(jnp.ones(K)))
            z = npd.Categorical(probs=w[clone])
        else:
            w = npy.sample("w", npd.Dirichlet(jnp.ones(K)))
            z = npd.Categorical(probs=w)

        q = npy.sample("q",
                       npd.TransformedDistribution(npd.LogNormal(loc=mu_q, scale=sigma_q).expand((K,)),
                                                   npd.transforms.OrderedTransform()))

        if x_neg is not None:
            with neg_sample_axis:
                if self.mode == "H":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=size_factor * q[0], concentration=alpha),
                                          obs=x_neg)
                elif self.mode == "C":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=size_factor * q[0],
                                                                concentration=alpha[clone]),
                                          obs=x_neg)
                else:
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=size_factor * q[0], concentration=alpha[0]),
                                          obs=x_neg)

        with sample_axis as i:

            # target pMHC
            if self.mode == "C":
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=(size_factor * q),
                                                                         concentration=alpha[clone].reshape(
                                                                             (sample_axis.size, 1))
                                                                         )
                                                )
            else:
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=size_factor * q,
                                                                         concentration=alpha))

            yhat = npy.sample("yhat", mixture, obs=x)

            # Until here, where we can track the membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))
