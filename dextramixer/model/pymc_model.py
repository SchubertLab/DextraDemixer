from typing import Dict, List, Optional, Tuple, Union
import abc

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from dextramixer.utils import RegisteredModel


class DextraMixer:
    """
    This class implements several mixture models to infer pMHC dextramere specificity from single cell immune profiling data
    with increasing usage of information

    **Given**:

    A read count matrix ***$X_{ij}\in \mathbb{N}$*** approximating the avidity of $i\in N$ T cell for the $j\in M$ epitope. The $N$ T cells can be grouped based on their T-cell receptor sequence into $C$ cluster.

    **Assumption**:

    1) Each read counts $X_{.j}$ of an epitope is iid.
    2) We assume that $X_{.j}$ can be represented as a mixture of two Negative Binomial distributions. Each clonal group $c \in C$ belongs to either the clone-specific **Binding** or the **Non-Binding** component. The **Non-Binding** component represents unspecific epitope binding and assay noise.
    3) It is assumed that unspecific binding T cells and non-binding T cells exhibit lower read counts compared to specifically binding T cell after appropriat normalization.
    4) T cells of a clonal group $c\in C$ are drawn from the same distribution.

    """

    @property
    def version(self):
        return self.model.version

    @property
    def model_type(self):
        return self.model.name

    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available epitope predictors and their supported versions

        :return: dict(str,list(str) - dictionary of epitope predictors represented as string and a list of supported
                                      versions
        """
        return [k for k in ADextraMixerModel.registry.keys()]

    def build_model(self,
                    X: Union[pd.Series, np.ndarray],
                    name: str = "mix",
                    mode: str = "H",
                    negCont: Union[pd.Series, np.ndarray] = None,
                    C: Union[pd.Series, np.ndarray] = None,
                    Sigma: Union[pd.Series, np.ndarray] = None,
                    **kwargs):
        """
        """
        self.trace = None
        self.model = ADextraMixerModel.registry.get(name, DextraMixerMixtureModel)()
        self.model.build_model(X, negCont, mode, C, Sigma)

    def data_setter(self,
                    X: Union[pd.Series, np.ndarray],
                    negCont: Union[pd.Series, np.ndarray] = None,
                    C: Union[pd.Series, np.ndarray] = None,
                    Sigma: Union[pd.DataFrame, np.ndarray] = None
                    ):
        self.model.data_setter(X, negCont, C, Sigma)

    def get_default_model_config(self) -> Dict[str, Union[int, float]]:
        return self.model.get_default_model_config

    @staticmethod
    def get_default_sampler_config() -> Dict[str, Union[int, float]]:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "nuts": {
                "target_accept": 0.95,
                "max_treedepth": 15
            }

        }
        return sampler_config

    def fit(self, sampler_config: Dict[str, Union[int, float]] = None) -> Union[
        az.InferenceData, pm.backends.base.MultiTrace]:
        """
        fits the hierarchical mixture model with MCMC and returns the trace
        """
        if self.model is None:
            raise Exception("Model is not initialized. Please call `build_model` first.")

        with self.model.model:
            if sampler_config is None:
                self.trace = pm.sample(**self.get_default_sampler_config())
            else:
                self.get_default_sampler_config().update(sampler_config)
                self.trace = pm.sample(**self.get_default_sampler_config())

        return self.trace

    def predict_posterior_class(self,
                                trace: Union[az.InferenceData, pm.backends.base.MultiTrace] = None,
                                rng: int = 42) -> Union[pd.Series, np.ndarray]:
        '''
        check papers for FDR control:
        https://academic.oup.com/bioinformatics/article/36/Supplement_2/i745/6055912
        https://www.jstor.org/stable/24775367
        https://hal.science/hal-03625469/document
        '''
        if self.model is None:
            raise Exception("Model is not initialized. Please call `build_model` first.")

        if trace is None:
            if self.trace is None:
                self.fit()
            trace = self.trace

        with self.model.posterior_predictive_model:
            pp = pm.sample_posterior_predictive(trace, var_names=[self.model.class_var], random_seed=rng)
            return pp.posterior_predictive[self.model.class_var]


class ADextraMixerModel(metaclass=RegisteredModel):
    """
    Abstract model class of DextraMixer
    """

    def build_model(self,
                    X: Union[pd.Series, np.ndarray],
                    negCont: Union[pd.Series, np.ndarray] = None,
                    mode: str = "H",
                    C: Union[pd.Series, np.ndarray] = None,
                    Sigma: Union[pd.Series, np.ndarray] = None,
                    **kwargs):
        """
        """
        self._generate_and_preprocess_model_data(X, negCont, mode, C, Sigma, **kwargs)
        self._build_inference_model(X,  negCont, mode, C, Sigma)
        self._build_posterior_predictive_model(X, negCont, mode, C, Sigma)

    @abc.abstractmethod
    def _build_inference_model(self,
                               X: Union[pd.Series, np.ndarray],
                               negCont: Union[pd.Series, np.ndarray],
                               mode: str = "H",
                               C: Union[pd.Series, np.ndarray] = None,
                               Sigma: Union[pd.Series, np.ndarray] = None,
                               **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_posterior_predictive_model(self,
                                          X: Union[pd.Series, np.ndarray],
                                          negCont: Union[pd.Series, np.ndarray] = None,
                                          mode: str = "H",
                                          C: Union[pd.Series, np.ndarray] = None,
                                          Sigma: Union[pd.Series, np.ndarray] = None,
                                          **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_default_model_config() -> Dict:
        return {}

    @property
    @abc.abstractmethod
    def class_var(self) -> str:
        return "z"

    @property
    @abc.abstractmethod
    def name(self) -> str:
        return "Abstract"

    @property
    @abc.abstractmethod
    def version(self) -> str:
        return "0.1"

    @property
    @abc.abstractmethod
    def model(self):
        return None

    @property
    @abc.abstractmethod
    def posterior_predictive_model(self):
        return None

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        """
        return self.get_default_model_config()

    def _generate_and_preprocess_model_data(self, X, negCont, mode, C, Sigma, **kwargs):

        if "model_config" in kwargs and isinstance(kwargs["model_config"], dict):
            self.model_config = DextraMixerMixtureModel.get_default_model_config().update(kwargs["model_config"])
        else:
            self.model_config = DextraMixerMixtureModel.get_default_model_config()

        self.model_coords: Dict = {
            "cluster": np.arange(2),
        }

        if C is not None:
            self.model_coords["clone"] = pd.unique(C)

        self.mode = mode

    def data_setter(self,
                    X: Union[pd.Series, np.ndarray],
                    negCont: Union[pd.Series, np.ndarray] = None,
                    C: Union[pd.Series, np.ndarray] = None,
                    Sigma: Union[pd.DataFrame, np.ndarray] = None
                    ):

        with self.model:
            pm.set_data({"x_data": X.values if isinstance(X, pd.Series) else X})

            if negCont is not None:
                pm.set_data({"negCont_data": negCont.values if isinstance(negCont, pd.Series) else negCont})
            if C is not None:
                pm.set_data({"c_data": C.values if isinstance(C, pd.Series) else C})
            if Sigma is not None:
                pm.set_data({"sigma_data": C.values if isinstance(Sigma, pd.DataFrame) else Sigma})


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
    differences in sequencing depth. $q_c$ is the expectation value of avidity for clone $c\in C$ or the expacation
    value of unspecific binding of epitope $i$.
    """

    __name = "mixture_model"
    __version = "1.0"
    __class_var = "z"
    _model = None
    _posterior_predictive_model = None

    @property
    def name(self):
        return self.__name

    @property
    def version(self):
        return self.__version

    @property
    def model(self):
        return self._model

    @property
    def class_var(self) -> str:
        return self.__class_var

    @property
    def posterior_predictive_model(self):
        return self._posterior_predictive_model

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            "mu_q_mean_prior": 0.0,
            "mu_q_var_prior": 10.0,
            "sigma_q_var_prior": 5.0,
            "alpha_var_prior": 10.0,
            "q_init_prior": [-5, 5]
        }
        return model_config

    def _build_inference_model(self, X: Union[pd.Series, np.ndarray],
                               negCont: Union[pd.Series, np.ndarray] = None, mode: str = "H",
                               C: Union[pd.Series, np.ndarray] = None, Sigma: Union[pd.Series, np.ndarray] = None,
                               **kwargs):

        with pm.Model(coords=self.model_coords) as self._model:

            # Create mutable data containers
            x_data = pm.MutableData("x_data", X)
            negCont_data = pm.MutableData("negCont_data", negCont)
            c_data = pm.MutableData("c_data", C)
            sigma_data = pm.MutableData("sigma_data", Sigma)

            # hyperprior parameters
            mu_q_mean_prior = self.model_config.get("mu_q_mean_prior", 0.0)
            mu_q_var_prior = self.model_config.get("mu_q_var_prior", 10.0)
            sigma_q_var_prior = self.model_config.get("sigma_q_var_prior", 5.0)
            alpha_var_prior = self.model_config.get("alpha_var_prior", 10.0)
            w_logits_prior = self.model_config.get("w_logits_prior", np.ones(2))
            q_init_prior = self.model_config.get("q_init_prior", [-5, 5])

            mu_q = pm.Normal("mu_q", mu_q_mean_prior, mu_q_var_prior)
            sigma_q = pm.HalfCauchy("sigma_q", sigma_q_var_prior)

            # shape prior
            if self.mode == "H":
                alpha = pm.HalfCauchy("alpha", alpha_var_prior)
            else:
                alpha = pm.HalfCauchy("alpha", alpha_var_prior, dims="cluster")
            pm.Potential("power_alpha", 1 / pm.math.sqrt(alpha))

            # cluster probability
            if C is not None:
                if Sigma is not None:
                    w_raw = pm.MvNormal("w_raw", 0, sigma_data)
                    w = pm.math.invlogit(w_raw)
                    w = pm.Deterministic("w", pm.math.stack([1 - w, w], axis=-1))

                else:
                    w = pm.Dirichlet("w", w_logits_prior, dims=("clone", "cluster"))
            else:
                w = pm.Dirichlet("w", w_logits_prior)

            # mean prior
            q_raw = pm.Normal("q_raw", mu_q, sigma_q, initval=q_init_prior,
                              transform=pm.distributions.transforms.univariate_ordered, dims="cluster")
            qs = pm.Deterministic("q", pm.math.exp(q_raw))

            components = pm.NegativeBinomial.dist(mu=qs, alpha=alpha)

            # likelihoods

            # negative control
            if negCont is not None:
                if mode == "H":
                    yhat = pm.NegativeBinomial("yhat_neg_control", mu=qs[0], alpha=alpha, observed=negCont_data)
                else:
                    yhat = pm.NegativeBinomial("yhat_neg_control", mu=qs[0], alpha=alpha[0], observed=negCont_data)

            if C is not None:
                yhat = pm.Mixture("yhat", w[c_data], components, observed=x_data)
            else:
                yhat = pm.Mixture("yhat", w, components, observed=x_data)

    def _build_posterior_predictive_model(self, X: Union[pd.Series, np.ndarray],
                                          negCont: Union[pd.Series, np.ndarray] = None, mode: str = "H",
                                          C: Union[pd.Series, np.ndarray] = None,
                                          Sigma: Union[pd.Series, np.ndarray] = None, **kwargs):

        with pm.Model(coords=self.model_coords) as self._posterior_predictive_model:

            # Create mutable data containers
            x_data = pm.MutableData("x_data", X)
            c_data = pm.MutableData("c_data", C)
            sigma_data = pm.MutableData("sigma_data", Sigma)

            # mean prior
            q_raw = pm.Normal("q_raw", shape=(2,))
            qs = pm.Deterministic("q", pm.math.exp(q_raw))

            # shape prior
            if self.mode == "H":
                alpha = pm.HalfCauchy("alpha", 1)
                comp_dists = [
                    pm.NegativeBinomial.dist(mu=qs[0], alpha=alpha),
                    pm.NegativeBinomial.dist(mu=qs[1], alpha=alpha)
                ]
            else:
                alpha = pm.HalfCauchy("alpha", 1, shape=(2,))
                comp_dists = [
                    pm.NegativeBinomial.dist(mu=qs[0], alpha=alpha[0]),
                    pm.NegativeBinomial.dist(mu=qs[1], alpha=alpha[1])
                ]

            # cluster probability
            if C is not None:
                if Sigma is not None:
                    w_raw = pm.MvNormal("w_raw", 0, sigma_data)
                    w = pm.math.invlogit(w_raw)
                    w = pm.Deterministic("w", pm.math.stack([1 - w, w], axis=-1))
                else:
                    w = pm.Dirichlet("w", np.ones(2),
                                     shape=(len(self.model_coords["clone"]),
                                            len(self.model_coords["cluster"])))

                log_probs = pm.math.concatenate([
                        [pm.math.log(w[c_data, 0]) + pm.logp(comp_dists[0], x_data)],
                        [pm.math.log(w[c_data, 1]) + pm.logp(comp_dists[1], x_data)]
                    ], axis=0)
            else:
                w = pm.Dirichlet("w", np.ones(2))
                log_probs = pm.math.concatenate([
                    [pm.math.log(w[0]) + pm.logp(comp_dists[0], x_data)],
                    [pm.math.log(w[1]) + pm.logp(comp_dists[1], x_data)],
                ], axis=0)

            z = pm.Categorical(self.class_var, logit_p=log_probs.T)
