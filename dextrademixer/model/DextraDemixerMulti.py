from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Union, Dict, Tuple, List, Iterable

import arviz as az
import jax
import jax.lax
import jax.numpy as jnp
import mudata as md
import numpy as np
import numpyro as npy
import pandas as pd
import tqdm
from jax import random, jit
from numpyro.infer.svi import SVIRunResult
from optax import exponential_decay

from dextrademixer.model import ApMHCDeconvolution
from dextrademixer.model.Dextrademixer import ADextraDemixerModel, DextraDemixer, DextraDemixerMixtureModel

if TYPE_CHECKING:
    pass

npy.enable_x64()

FLOAT_DTYPE = "float64"
INT_DTYPE = "int32"

class DextraDemixerMulti(DextraDemixer):
    r"""
    Fit DextraDemixer independently to multiple pMHC features.

    Parameters
    ----------
    model_type : str, default="mixturemodel"
        Registered probabilistic model implementation.
    mode : {"I"}, default="I"
        Multi-pMHC fitting mode. Only independent fitting is supported.
    alpha_model : str, default="overdispersion"
        Parameterization used for negative-binomial dispersion.

    Notes
    -----
    This experimental implementation attempts to reuse compiled inference
    machinery across pMHCs with equal input dimensions. It is not part of the
    supported package API.
    """

    def __init__(self, model_type: str = "mixturemodel", mode: str = "I", alpha_model="overdispersion"):
        super().__init__()

        if mode.upper() not in ("I"):
            raise ValueError(f"`mode` must be either of the three `I`=independent, `H`=hierarchical, "
                             + f"`C`=clonotype-specific but was {mode}")

        if model_type not in ADextraDemixerModel.registry.keys():
            raise warnings.warn(f"`model_type` {model_type} not supported using the standard model.")


        #technical variables
        self.rng_key = None
        self.mode = mode.upper()
        self.alpha_model = alpha_model

        self.traces = None
        self.svi_results = None

        self.model = ADextraDemixerModel.registry.get(model_type, DextraDemixerMixtureModel)()
        self.sampler = None
        self.svi = None
        self.optimizer = None
        self.guides = None
        self.is_svi = None

        # input data
        self.pmhc_names = None
        self.N = None
        self.M = None
        self.x = None
        self.x_neg = None
        self.s = None
        self.c = None
        self.sigma = None

    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_keys: List[str],
                              gex_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_key: str = None,
                              ir_cov_key: str = None,
                              **kwargs) -> None:
        """
        Preprocess multiple pMHC features and initialize model state.

        Parameters
        ----------
        mdata : mudata.MuData
            Cell-aligned count and immune-receptor modalities.
        pmhc_keys : list of str
            Target pMHC features.
        gex_key : str, default="gex"
            Count-modality key.
        neg_ctrl_key : str, optional
            Negative-control feature.
        ir_key : str, default="airr"
            Immune-receptor modality key.
        ir_clone_key : str, optional
            Observation column containing clonotype IDs.
        ir_cov_key : str, optional
            Key containing a clonotype covariance matrix.
        **kwargs : object
            Additional model-specific values.
        """
        gex = mdata.mod[gex_key]
        air = mdata.mod[ir_key]

        # extract data specific information
        if neg_ctrl_key in pmhc_keys:
            pmhc_keys.remove(neg_ctrl_key)
        self.M = len(pmhc_keys)
        self.N = gex.shape[0]
        self.pmhc_names = pmhc_keys

        self.c = jnp.array(air.obs[ir_clone_key].to_numpy().astype("int32")) if ir_clone_key is not None else None
        self.sigma = jnp.array(air.uns[ir_cov_key]) if ir_cov_key is not None else None

        if self.mode == "C":
            if self.c is None:
                raise ValueError("If `mode`= C a clonotype vector `c` must be specified.")

        if len(pmhc_keys) > 1:
            x_plus = jnp.array(gex[:, pmhc_keys + [neg_ctrl_key]].X.toarray(),
                               dtype=FLOAT_DTYPE)  # only used for size factor calculation
            s = self.__size_factors(x_plus)
            del x_plus
        else:
            s = jnp.ones(self.N, dtype=FLOAT_DTYPE)
        self.s = s

        self.x = jnp.array(gex[:, pmhc_keys].X.toarray(), dtype=FLOAT_DTYPE)
        self.x_neg = jnp.array(gex[:, neg_ctrl_key].X.toarray().reshape((self.N,)),
                          dtype=FLOAT_DTYPE) if neg_ctrl_key else None

        self._check_parameters(self.x, self.x_neg, self.c, self.sigma)

        # technical variables:
        self.traces = [None] * self.M
        self.svi_results = [None] * self.M
        self.guides = [None] * self.M

    @staticmethod
    def __size_factors(counts: jnp.ndarray) -> jnp.ndarray:
        """
        DEGSeq2 size factor calculation
        """

        log_counts = jnp.log(counts)
        log_counts = jnp.where(jnp.isinf(log_counts), jnp.nan, log_counts)
        log_means = jnp.nanmean(log_counts, axis=0)

        mask = log_means > 0 # TODO not sure this is correct
        log_ratios = log_counts[:, mask] - log_means[mask]
        log_medians = jnp.nanmedian(log_ratios, axis=1)

        return jnp.exp(log_medians)

    def fit(self, sampler_config: Dict[str, Union[int, float]] = None, rng_key: int = 3) -> None:
        """
        Fit every pMHC model with Markov chain Monte Carlo.

        Parameters
        ----------
        sampler_config : dict, optional
            MCMC and NUTS configuration overrides.
        rng_key : int, default=3
            Random seed.

        Notes
        -----
        Fitted traces are stored in :attr:`traces`.
        """
        if self.x is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.is_svi = False

        if sampler_config is None:
            sampler_config = self.get_default_sampler_config()["mcmc"]

        nuts_config = {**self.get_default_sampler_config()["mcmc"]["nuts"], **sampler_config.get("nuts", {})}
        sampling_config = {**self.get_default_sampler_config()["mcmc"], **sampler_config}
        sampling_config.pop("nuts", None)


        for j in range(self.M):
            # preprocess model with new incoming data
            self.model.preprocess_model_data(x=self.x[:,j], s=self.s, neg_cont=self.x_neg, c=self.c, sigma=self.sigma,
                                             alpha_model=self.alpha_model, mode=self.mode)
            self.__fit(j, nuts_config, sampling_config, rng_key)
        return self.traces

    def __fit(self,
              j: int,
              nuts_config: Dict[str, Union[int, float]],
              sampling_config: Dict[str, Union[int, float]],
              rng_key: int) -> None:

        if sampling_config["progress_bar"]:
            print(f"Fitting {j + 1}. pMHC:\n", file=sys.stderr)

        if self.sampler is None:
            self.sampler = npy.infer.MCMC(
                npy.infer.NUTS(self.model.model, **nuts_config),
                **sampling_config
            )

        self.sampler.run(random.PRNGKey(rng_key))
        self.traces[j] = az.from_numpyro(self.sampler)


    def fit_svi(self, guide=npy.infer.autoguide.AutoMultivariateNormal, svi_config: Dict[str, Union[int, float]] = None,
                nof_inits: int = 100, use_minimal_loss: bool = True, rng_key: int = 998777,
                return_loss: bool = False) -> List[az.InferenceData]:
        """
        Fit every pMHC model with stochastic variational inference.

        Parameters
        ----------
        guide : type
            NumPyro autoguide constructor.
        svi_config : dict, optional
            Optimizer and SVI configuration overrides.
        nof_inits : int, default=100
            Number of random initializations.
        use_minimal_loss : bool, default=True
            Retain parameters from the lowest-loss iteration.
        rng_key : int, default=998777
            Random seed.
        return_loss : bool, default=False
            Reserved compatibility option.

        Returns
        -------
        list of arviz.InferenceData
            One inference trace per pMHC.
        """

        if self.x is None:
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

        self.optimizer = npy.optim.ClippedAdam(adam_config["init_value"])

        for j in range(self.M):
            # preprocess model with new incoming data
            self.model.preprocess_model_data(x=self.x[:,j], s=self.s, neg_cont=self.x_neg, c=self.c, sigma=self.sigma,
                                             alpha_model=self.alpha_model, mode=self.mode)
            self.__fit_svi(j, guide, tracer_config, svi_config, nof_inits, use_minimal_loss, rng_key)
        return self.traces

    def __fit_svi(self,
                  j: int,
                  guide: type[npy.infer.autoguide.AutoGuide],
                  tracer_config: Dict[str, Union[int, float]],
                  svi_config: Dict[str, Union[int, float]],
                  nof_inits: int, use_minimal_loss: bool,
                  rng_key: int) -> None:
        """
        Fit one pMHC with stochastic variational inference.

        Parameters
        ----------
        j : int
            pMHC index.
        guide : type
            NumPyro autoguide constructor.
        tracer_config : dict
            ELBO tracer configuration.
        svi_config : dict
            Optimizer and iteration configuration.
        nof_inits : int
            Number of random initializations.
        use_minimal_loss : bool
            Retain parameters from the lowest-loss iteration.
        rng_key : int
            Random seed.
        """

        # find good random initialization
        random_init = []
        for i, key in enumerate(random.split(random.PRNGKey(rng_key), nof_inits)):
            if callable(getattr(self.model, "guide", None)):
                local_guide = self.model.guide
            else:
                local_guide = guide(self.model.model, init_loc_fn=npy.infer.initialization.init_to_median)
            svi = npy.infer.SVI(self.model.model, local_guide, self.optimizer,
                                loss=npy.infer.TraceGraph_ELBO(**tracer_config))
            init_state = svi.init(key)
            loss = svi.evaluate(init_state)

            # Initialization depends on the guide, so need to save the best guide
            random_init.append((loss, key, local_guide))

        init_losses = np.array([x[0] for x in random_init])
        best_idx = jnp.nanargmin(init_losses)
        best_loss, best_key, best_guide = random_init[best_idx]

        self.guides[j] = best_guide
        svi = npy.infer.SVI(self.model.model, self.guides[j], self.optimizer,
                            loss=npy.infer.TraceGraph_ELBO(**tracer_config))

        def body_fn(svi_state, step):
            svi_state, loss = svi.stable_update(svi_state, step=step)
            return svi_state, loss, svi.get_params(svi_state)

        svi_state = svi.init(rng_key=best_key)
        losses = []
        params = []
        with tqdm.trange(1, svi_config.get("maxiter", 1000) + 1,
                         desc=f"Fitting {j + 1}. pMHC: ",
                         disable=(not svi_config.get("progress_bar", False))) as t:
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

        params = params[jnp.argmin(losses)] if use_minimal_loss else params[-1]
        svi_result = SVIRunResult(params=params, losses=losses, state=svi_state)
        self.svi_results[j] = svi_result

        posterior_samples = self.guides[j].sample_posterior(random.PRNGKey(self.rng_key), svi_result.params,
                                                        sample_shape=(500,))

        # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
        posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
        self.traces[j] = az.from_dict(posterior=posterior_samples_np)

    def predict_posterior_class(self,
                                max_pmhc=False,
                                clone_majority=False,
                                threshold: Union[List[float], float] = None,
                                target_fdr: Union[List[float], float] = None,
                                quantile: Union[List[float], float] = None,
                                cred_intvl: Union[List[float], float] = None,
                                clonotype_adherence: Union[List[bool], bool] = False
                                ) -> Tuple[np.array, np.array]:
        """
        Predict binding probabilities and assignments for every pMHC.

        On a global level two summarization strategies can be combined to generate unique assignments per cell.
        1) max posterior class probability across all pMHCs combined with threshold or target_fdr assignment and
        2) majority pMHC assignment per clonotype (if such information is provided).
        Both approaches can be combined, applying first max posterior class assignment then majority pMHC class
        assignment. Ties will not be resolved.

        Parameters
        ----------
        max_pmhc : bool, default=False
            Retain only the pMHC with the highest probability per cell.
        clone_majority : bool, default=False
            Retain the majority pMHC assignment within each clonotype.
        threshold : float or list of float, optional
            Fixed probability threshold for each pMHC.
        target_fdr : float or list of float, optional
            Target Bayesian false discovery rate for each pMHC.
        quantile : float or list of float, optional
            Lower posterior quantile used instead of the mean.
        cred_intvl : float or list of float, optional
            Required posterior probability of FDR control.
        clonotype_adherence : bool or list of bool, default=False
            Use clonotype probability vectors when available.

        Returns
        -------
        p : numpy.ndarray
            Cell-by-pMHC binding probabilities.
        assignment : numpy.ndarray
            Cell-by-pMHC binary assignments.
        """

        def __check_input(input, er_msg):
            if isinstance(input, Iterable):
                if len(input) != self.M:
                    raise ValueError(er_msg.format(self.M, len(input)))
            else:
                input = [input] * self.M
            return input

        def __return_p_summary(p_sample, _quantile=None, _cred_intvl=None):
            if _quantile:
                return jnp.quantile(p_sample, _quantile, axis=0)[:, 1]
            elif _cred_intvl:
                return p_sample
            else:
                return jnp.nanmean(p_sample, axis=0)[:, 1]

        if self.is_svi is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        threshold = __check_input(threshold,
                                  "`threshold` must be a float or a list of length {} but has length {}.")
        target_fdr = __check_input(target_fdr,
                                   "`target_fdr` must be a float or a list of length {} but has length {}.")
        clonotype_adherence = __check_input(clonotype_adherence,
                                   "`clonotype_adherence` must be a float or a list of length {} but has length {}.")
        quantile = __check_input(quantile,
                                            "`quantile` must be a bool or a list of length {} but has length {}.")
        cred_intvl = __check_input(cred_intvl,
                                            "`cred_intvl` must be a bool or a list of length {} but has length {}.")

        ps, assignments = [], []

        for j in range(self.M):
            # posterior probability of belonging to the binding class
            if self.is_svi:
                if clonotype_adherence[j] and self.model.data["clone_continuous"] is not None:
                    posterior_samples = self.guides[j].sample_posterior(random.PRNGKey(self.rng_key),
                                                                        self.svi_results[j].params,
                                                                        sample_shape=(500,))

                    # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
                    p = __return_p_summary(jnp.array(posterior_samples["w"]), quantile[j], cred_intvl[j])
                else:
                    predictive = npy.infer.Predictive(self.model.model,
                                                      guide=self.guides[j],
                                                      params=self.svi_results[j].params,
                                                      num_samples=500)
                    samples = predictive(jax.random.PRNGKey(self.rng_key)) # self.rng_key
                    p = __return_p_summary(jnp.exp(samples["log_p"]), quantile[j], cred_intvl[j])

            else:
                if clonotype_adherence[j] and self.model.data["clone_continuous"] is not None:
                    w = jnp.array(self.traces[j].posterior["w"])
                    # requires chain flattening to be compatible with svi-branch
                    w = w.reshape((w.shape[0] * w.shape[1],) + w.shape[2:])
                    p = __return_p_summary(w, quantile[j], cred_intvl[j])
                else:
                    log_p = jnp.array(self.traces[j].posterior["log_p"].values)
                    # requires chain flattening to be compatible with svi-branch
                    log_p = log_p.reshape((log_p.shape[0] * log_p.shape[1],) + log_p.shape[2:])
                    p = __return_p_summary(jnp.exp(log_p[..., [0, 1]]), quantile[j], cred_intvl[j])

            if cred_intvl[j] is not None:
                p, assignment, threshold = self._predict_posterior_class_dist(p, target_fdr[j], cred_intvl[j])
            else:
                assignment = self._predict_posterior_class(p, threshold[j], target_fdr[j])

            if clonotype_adherence[j] and self.model.data["clone_continuous"] is not None:
                assignment = assignment[self.model.data["clone_continuous"]]
                p = p[self.model.data["clone_continuous"]]

            ps.append(p)
            assignments.append(assignment)

        ps, assignments = np.vstack(ps).T, np.vstack(assignments).T

        # max p assignment per cell
        if max_pmhc:
            assignments = ((ps == ps.max(axis=1, keepdims=True)) & assignments.astype(bool)).astype(int)

        # clonal majority assignment
        if clone_majority and self.model.data["clone_continuous"] is not None:
            c = self.model.data["clone_continuous"]
            tmp = np.zeros_like(assignments)

            for g in np.unique(c):
                rows = np.where(c == g)[0]
                col_counts = assignments[rows].sum(axis=0)
                max_count = col_counts.max()
                tmp[np.ix_(rows, np.where(col_counts == max_count)[0])] = 1 if max_count > 0 else 0
            assignments = tmp

        return ps, assignments

    def summary(self):
        """
        Summarize posterior parameters for every fitted pMHC.

        Returns
        -------
        pandas.DataFrame
            ArviZ summaries indexed by pMHC.
        """
        summaries = []
        keys = []
        for j in range(self.M):
            summaries.append(az.summary(self.traces[j], var_names=["~log_p"]))
            keys.append(self.pmhc_names[j])
        return pd.concat(summaries, keys=keys, names=["pMHC"])
