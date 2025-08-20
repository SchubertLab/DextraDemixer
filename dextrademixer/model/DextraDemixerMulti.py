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

    def __init__(self, model_type: str = "mixturemodelkmeans", mode: str = "I"):
        super().__init__()

        if mode.upper() not in ("I"):
            raise ValueError(f"`mode` must be either of the three `I`=independent, `H`=hierarchical, "
                             + f"`C`=clonotype-specific but was {mode}")

        if model_type not in ADextraDemixerModel.registry.keys():
            raise warnings.warn(f"`model_type` {model_type} not supported using the standard model.")

        self.N = None
        self.M = None
        self.traces = None
        self.is_svi = None
        self.svi_results = None
        self.rng_key = None
        self.mode = mode.upper()
        self.guides = None
        self.models = None
        self.samplers = None

    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_keys: List[str],
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
            pmhc_keys: a list of strings specifying the pMHC columns in `gex_key` modality`s `X` which should be deconvolved
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
        self.N =  gex.shape[0]

        if neg_ctrl_key in pmhc_keys:
            pmhc_keys.remove(neg_ctrl_key)
        self.M = len(pmhc_keys)

        self.traces = [None]*self.M
        self.svi_results = [None]*self.M
        self.guides = [None]*self.M
        self.models = [None]*self.M
        self.samplers = [None]*self.M

        if len(pmhc_keys) > 1:
            x_plus = jnp.array(gex[:, pmhc_keys + [neg_ctrl_key]].X.toarray(), dtype=FLOAT_DTYPE) # only used for size factor calculation
            s = self.__size_factors(x_plus)
            del x_plus
        else:
            s = jnp.ones(self.N, dtype=FLOAT_DTYPE)

        x = jnp.array(gex[:, pmhc_keys].X.toarray(), dtype=FLOAT_DTYPE)
        x_neg = jnp.array(gex[:, neg_ctrl_key].X.toarray().reshape((self.N,)), dtype=FLOAT_DTYPE) if neg_ctrl_key else None

        c = jnp.array(air.obs[ir_clone_key].to_numpy().astype("int32")) if ir_clone_key is not None else None
        sigma = jnp.array(air.uns[ir_cov_key]) if ir_cov_key is not None else None

        if self.mode == "C":
            if c is None:
                raise ValueError("If `mode`= C a clonotype vector `c` must be specified.")

        self._check_parameters(x, x_neg, c, sigma)

        # initialize individual models
        for j in range(x.shape[1]):
            model = ADextraDemixerModel.registry.get(self.model_type, DextraDemixerMixtureModel)()
            model.preprocess_model_data(x[:,j], s, x_neg, c, sigma, self.mode, **kwargs)
            self.models[j] = model

    @staticmethod
    def __size_factors(counts: jnp.ndarray) -> jnp.ndarray:
        """
        DEGSeq2 size factor calculation
        """

        log_counts = jnp.log(counts)
        log_counts = jnp.where(jnp.isinf(log_counts), jnp.nan, log_counts)
        log_means = jnp.nanmean(log_counts, axis=0)

        mask = log_means > 0
        log_ratios = log_counts[:, mask] - log_means[mask]
        log_medians = jnp.nanmedian(log_ratios, axis=1)

        return jnp.exp(log_medians)

    def fit(self, sampler_config: Dict[str, Union[int, float]] = None, rng_key: int = 3)-> List[az.InferenceData]:
        """
        fits the mixture model with MCMC and returns the trace
        """
        if len(self.models) == 0:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.is_svi = False

        if sampler_config is None:
            sampler_config = self.get_default_sampler_config()["mcmc"]

        nuts_config = {**self.get_default_sampler_config()["mcmc"]["nuts"], **sampler_config.get("nuts", {})}
        sampling_config = {**self.get_default_sampler_config()["mcmc"], **sampler_config}
        sampling_config.pop("nuts", None)
        return [self.__fit(j, nuts_config, sampling_config, rng_key) for j in range(len(self.models))]

    def __fit(self,
              j: int,
              nuts_config: Dict[str, Union[int, float]],
              sampling_config: Dict[str, Union[int, float]],
              rng_key: int) -> az.InferenceData:

        if sampling_config["progress_bar"]:
            print(f"Fitting {j+1}. pMHC:\n", file=sys.stderr)

        sampler = npy.infer.MCMC(
            npy.infer.NUTS(self.models[j].model, **nuts_config),
            **sampling_config
        )

        sampler.run(random.PRNGKey(rng_key))

        trace = az.from_numpyro(sampler)
        self.traces[j] = trace
        self.samplers[j] = sampler

        return trace

    def fit_svi(self, guide=npy.infer.autoguide.AutoMultivariateNormal, svi_config: Dict[str, Union[int, float]] = None,
                nof_inits: int = 100, use_minimal_loss: bool = True, rng_key: int = 998777)->List[az.InferenceData]:

        if len(self.models) == 0:
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

        return [self.__fit_svi(j, guide, adam_config, tracer_config, svi_config, nof_inits, use_minimal_loss)
                for j in range(len(self.models))]

    def __fit_svi(self,
                  j: int,
                  guide: type[npy.infer.autoguide.AutoGuide],
                  adam_config: Dict[str, Union[int, float]] ,
                  tracer_config: Dict[str, Union[int, float]],
                  svi_config: Dict[str, Union[int, float]],
                  nof_inits: int, use_minimal_loss: bool) -> az.InferenceData:
        """
        Implements stochastic variational inference

        j: index of pmhc
        guide: The guide to use for variational inference. If None, self.model object will be checked for a guide function
        svi_config: configuration for optimizer (Adam) and posterior samples
        nof_inits: number of initializations tried with different seeds to find gut init values
        use_minimal_loss: boolean indicating whether to report the parameters with the lowest loss instead
        """

        model = self.models[j]

        # check for custom guide in self.model otherwise use autoguide
        if callable(getattr(model, "guide", None)):
            guide = model.guide
        else:
            guide = guide(model.model)

        self.guides[j] = guide

        adam_config["transition_steps"] = svi_config.get("maxiter", 1000) // 2

        optimizer = npy.optim.ClippedAdam(exponential_decay(**adam_config))
        svi = npy.infer.SVI(model.model, guide, optimizer, loss=npy.infer.TraceGraph_ELBO(**tracer_config))

        # find good random initialization
        best_loss, best_key = min((svi.evaluate(svi.init(key)), key) for key in
                                  random.split(random.PRNGKey(self.rng_key), nof_inits))

        def body_fn(svi_state, step):
            svi_state, loss = svi.stable_update(svi_state, step=step)
            return svi_state, loss, svi.get_params(svi_state)

        svi_state = svi.init(rng_key=best_key)
        losses = []
        params = []
        with tqdm.trange(1, svi_config.get("maxiter", 1000) + 1,
                         desc=f"Fitting {j+1}. pMHC: ",
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

        posterior_samples = guide.sample_posterior(random.PRNGKey(self.rng_key), svi_result.params,
                                                        sample_shape=(500,))

        # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
        posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
        inference_data = az.from_dict(posterior=posterior_samples_np)

        return inference_data

    def predict_posterior_class(self,
                                max_pmhc = False,
                                clone_majority = False,
                                threshold: Union[List[float], float] = None,
                                target_fdr: Union[List[float], float] = None,
                                clonotype_adherence: Union[List[bool], bool] = False
                                ) -> Tuple[np.array, np.array]:
        """
        Returns the binder assignments based on the inferred posterior class probabilities.
        Assignment can be either be done by providing a threshold or target fdr value if FDR control is wanted.
        If neither threshold nor target_fdr is provided the max posterior class probability will be used.

        On a global level two summarization strategies can be combined to generate unique assignments per cell.
        1) max posterior class probability across all pMHCs combined with threshold or target_fdr assignment and
        2) majority pMHC assignment per clonotype (if such information is provided).
        Both approaches can be combined, applying first max posterior class assignment then majority pMHC class
        assignment. Ties will not be resolved.

        Args:
             max_pmhc: whether to report the pMHC with highest posterior probability across all pMHCs as sole assignment
                       in case of ties all possible pMHCs will be assigned
             clone_majority: whether to report the majority pMHC as sole assignment for cells of a clonotype
                           (can be used in combinatio with max_pmhc)
             threshold: (Optional) a threshold in [0,1] determining binder based on inferred posterior class
                        probabilities
            target_fdr: (Optional) the FDR threshold to control False discovery rate based on the posterior
                        class probability
            clonotype_adherence: instead of using posterior class assignment per cell use clonotype probability vector
                                if available.
        Returns:
            A tuple (p, assignment) of arrays with p being the posterior probability of binding and assignment the
            class assignment decision
        """
        def __check_input(input, er_msg):
            if isinstance(input, Iterable):
                if len(input) != self.M:
                    raise ValueError(er_msg.format(self.M, len(input)))
            else:
                input = [input] * self.M
            return input

        if self.is_svi is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        threshold = __check_input(threshold,
                                  "`threshold` must be a float or a list of length {} but has length {}.")
        target_fdr = __check_input(target_fdr,
                                   "`target_fdr` must be a float or a list of length {} but has length {}.")
        clonotype_adherence = __check_input(clonotype_adherence,
                                            "`clonotype_adherence` must be a bool or a list of length {} but has length {}.")

        ps, assignments = [], []

        for j in range(self.M):
            # posterior probability of belonging to the binding class
            if self.is_svi:
                if clonotype_adherence and self.models[j].data["clone_continuous"] is not None:
                    # TODO ALTERNATIVE USE MAJORITY VOTING IN CLONE?
                    posterior_samples = self.guides[j].sample_posterior(random.PRNGKey(self.rng_key),
                                                                    self.svi_result[j].params,
                                                                    sample_shape=(500,))

                    # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
                    p = jnp.nanmean(posterior_samples["w"], axis=0)[:, 1]
                else:
                    predictive = npy.infer.Predictive(self.models[j].model,
                                                      guide=self.guides[j],
                                                      params=self.svi_results[j].params,
                                                      num_samples=500)
                    samples = predictive(jax.random.PRNGKey(self.rng_key))  # self.rng_key
                    p = jnp.exp(jnp.nanmean(samples["log_p"], axis=0))[:, 1]

            else:
                if clonotype_adherence[j] and self.models[j].data["clone_continuous"] is not None:
                    p = jnp.mean(self.samplers[j].get_samples()["w"], axis=0)[:, 1]
                else:
                    p = jnp.mean(jnp.exp(self.samplers[j].get_samples()["log_p"][..., 1]), axis=0)

            assignment = self._predict_posterior_class(p, threshold[j], target_fdr[j])

            if clonotype_adherence[j] and self.models[j].data["clone_continuous"] is not None:
                assignment = assignment[self.model[j].data["clone_continuous"]]
                p = p[self.models[j].data["clone_continuous"]]

            ps.append(p)
            assignments.append(assignment)

        ps, assignments = np.vstack(ps).T, np.vstack(assignments).T

        # max p assignment per cell
        if max_pmhc:
            assignments = ((ps == ps.max(axis=1, keepdims=True)) & assignments.astype(bool)).astype(int)

        # clonal majority assignment
        if clone_majority and self.models[0].data["clone_continuous"] is not None:
            c = self.models[0].data["clone_continuous"]
            tmp = np.zeros_like(assignments)

            for g in np.unique(c):
                rows = np.where(c == g)[0]
                col_counts = assignments[rows].sum(axis=0)
                max_count = col_counts.max()
                tmp[np.ix_(rows, np.where(col_counts == max_count)[0])] = 1 if max_count > 0 else 0
            assignments = tmp

        return ps, assignments

    def summary(self):
        raise NotImplementedError()