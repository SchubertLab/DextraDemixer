from __future__ import annotations

import abc

from typing import TYPE_CHECKING, Literal, Optional, Union, Dict

import arviz as az
import numpy as np
import pandas as pd

import scanpy as sc
# import scirpy as ir
# from mudata import MuData
from anndata import AnnData

from jax import random
from jax.nn import logsumexp
from jax.config import config

import jax.numpy as jnp

import numpyro as npy
import numpyro.distributions as npd
from numpyro.infer import HMC, MCMC, NUTS, initialization

from dextramixer.utils import RegisteredModel

if TYPE_CHECKING:
    from jax._src.prng import PRNGKeyArray
    from jax._src.typing import Array

config.update("jax_enable_x64", True)


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
       specifically binding T cell after appropriat normalization.
    4) T cells of a clonal group $c\in C$ are drawn from the same distribution.

    """

    def __init__(self, model_type: str = "mixturemodel", mode: str = "H", ):
        self.sampler = None
        self.trace = None
        self.mode = mode
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
                              x: Union[pd.Series, np.ndarray, Array],
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[pd.Series, np.ndarray, Array] = None,
                              **kwargs):

        self.model.preprocess_model_data(x, neg_cont, c, sigma, self.mode, **kwargs)

    def get_default_model_config(self) -> Dict[str, Union[int, float]]:
        return self.model.get_default_model_config

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
            raise Exception("Model is not initialized. Please call `build_model` first.")

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

    def predict_posterior_class(self, threshold: float = None, target_fdr: float = None):

        if threshold is None and target_fdr is None:
            raise ValueError("Either a manual `threshold` or a `target_fdr` must be specified.")
        if threshold is not None and target_fdr is not None:
            raise ValueError("Please specify either a manual `threshold` or a `target_fdr` but not both.")

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
        assignment = (p >= threshold).astype("int32")
        return p, assignment

    def __make_arvis(self):
        self.trace = az.from_numpyro(self.sampler)
        return self.trace


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
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              **kwargs):
        """
        """
        float_dtype = "float64"
        int_dtype = "int64"

        self.data = {"x": jnp.array(x, dtype=float_dtype),
                     "neg_cont": None if neg_cont is None else jnp.array(neg_cont, dtype=float_dtype),
                     "clone": None if c is None else jnp.array(c, dtype=int_dtype),
                     "sigma": None if sigma is None else jnp.array(sigma, dtype=float_dtype),
                     }

        self.mode = mode

        N = x.shape[0]
        K = 2

        # set coord axis
        self.coords = {
            "sample_axis": npy.plate("sample_axis", N, dim=-1),
            "cluster_axis": npy.plate("cluster_axis", K, dim=-1),
        }

        if self.data["clone"] is not None:
            C_nof = len(jnp.unique(self.data["clone"]))
            self.coords["clone_axis"] = npy.plate("clone_axis", C_nof, dim=-1)

        if self.data["neg_cont"] is not None:
            N_neg = neg_cont.shape[0]
            self.coords["neg_sample_axis"] = npy.plate("neg_sample_axis", N_neg, dim=-1)

        if self.data["sigma"] is not None:
            if self.data["clone"] is None:
                raise ValueError("If `sigma` is given, clonality vector `c` must be given as well")
            else:
                if self.data["sigma"].shape[0] != C_nof:
                    raise ValueError(("Sigma must have shape ({C_nof},{C_nof}) and defined over clonotypes but has"
                                      + "{sigma_shape}").format(C_nof=C_nof, sigma_shape=self.data["sigma"].shape))

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
    &\frac{X_{.j}}{s_i}\sim (1-w^0_{C(i)})\text{NegBinom}(e^{q^0}, \alpha) + w^1_{C(i)}\text{NegBinom}(e^{q^1}, \alpha)&\forall i \in N\\
    \text{ELSE:}\\
    &\frac{X_{.j}}{s_i}\sim (1-w^0_{C(i)})\text{NegBinom}(e^{q^0}, \alpha^0) + w^1_{C(i)}\text{NegBinom}(e^{q^1}, \alpha^1)&\forall i \in N\\
    \text{ELSE:}\\
    &w \sim \text{Dir}([1,1])\\
    \text{IF mode == "H":}\\
    &\frac{X_{.j}}{s_i}\sim (1-w^0)\text{NegBinom}(e^{q^0}, \alpha) + w^1\text{NegBinom}(e^{q^1}, \alpha)&\forall i \in N\\
    \text{ELSE:}\\
    &\frac{X_{.j}}{s_i}\sim (1-w^0)\text{NegBinom}(e^{q^0}, \alpha^0) + w^1\text{NegBinom}(e^{q^1}, \alpha^1)&\forall i \in N\\
    \text{IF $X_{.j}^{\text{neg}}$ defined:}\\
    \text{IF mode == "H":}\\
    &X_{.j}^{\text{neg}} \sim \text{NegBinom}(e^{q^0}, \alpha)&\forall i \in N\\
    \text{ELSE:}\\
    &X_{.j}^{\text{neg}} \sim \text{NegBinom}(e^{q^0}, \alpha^0)&\forall i \in N\\
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
            raise RuntimeError("Model was not properly initialized. Please call first `preprocess_model_data`")

        model_config = {**DextraMixerMixtureModel.get_default_model_config(), **kwargs["model_config"]} if (
                "model_config" in kwargs) else DextraMixerMixtureModel.get_default_model_config()

        # data
        x = self.data["x"]
        clone = self.data["clone"]
        sigma = self.data["sigma"]
        neg_cont = self.data["neg_cont"]

        # plates
        sample_axis = self.coords["sample_axis"]
        cluster_axis = self.coords["cluster_axis"]
        K = cluster_axis.size
        if clone is not None:
            clone_axis = self.coords["clone_axis"]
        if neg_cont is not None:
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

        q = npy.sample("q", npd.TransformedDistribution(npd.LogNormal(loc=mu_q, scale=sigma_q).expand([K]),
                                                        npd.transforms.OrderedTransform()))
        mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=q, concentration=alpha))

        if neg_cont is not None:
            with neg_sample_axis:
                if self.mode == "H":
                    yhat_neg = npy.sample("yhat_neg", npd.NegativeBinomial2(mean=q[0], concentration=alpha), obs=neg_cont)
                else:
                    yhat_neg = npy.sample("yhat_neg", npd.NegativeBinomial2(mean=q[0], concentration=alpha[0]),
                                          obs=neg_cont)

        with sample_axis:
            yhat = npy.sample("yhat", mixture, obs=x)

            # Until here, where we can track the membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))
