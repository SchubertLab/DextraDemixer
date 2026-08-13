from __future__ import annotations

import abc
import warnings
import os
import pickle

from typing import Any, Self, Sequence, Union, Dict, Tuple

import arviz as az
import tqdm
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
from optax import exponential_decay

from dextrademixer.model import ApMHCDeconvolution
from dextrademixer.utils import RegisteredModel, calculate_metrics

npy.enable_x64()

FLOAT_DTYPE = "float64"
INT_DTYPE = "int32"


class DextraDemixer(ApMHCDeconvolution):
    r"""
    Infer pMHC dextramer specificity with a negative-binomial mixture model.

    Parameters
    ----------
    model_type : str, default="mixturemodelkmeans"
        Registered model implementation to use.
    model_config : dict, optional
        Values that override the selected model's default configuration.

    Attributes
    ----------
    sampler : object or None
        NumPyro sampler used for inference, when applicable.
    trace : arviz.InferenceData or None
        Inference trace produced by a sampler.
    svi_result : numpyro.infer.svi.SVIRunResult or None
        Result of stochastic variational inference.
    model : ADextraDemixerModel
        Selected probabilistic model implementation.

    Notes
    -----
    Given a read-count vector :math:`X_{ij} \in \mathbb{N}` for cells
    :math:`i` and epitopes :math:`j`, the model assumes two
    negative-binomial components representing binding and non-binding cells.
    Cells belonging to the same clonotype may share probability summaries,
    and the binding component is constrained to have higher normalized counts.
    """

    def __init__(self, model_type: str = "mixturemodelkmeans",
                 model_config: dict[str, Any] | None = None):
        super().__init__()

        self.sampler = None
        self.trace = None
        self.svi_result = None
        self.rng_key = None
        self.guide = None
        self.model_config = model_config if model_config is not None else {}

        if model_type not in ADextraDemixerModel.registry.keys():
            raise warnings.warn(f"`model_type` {model_type} not supported using the standard model.")
        self.model = ADextraDemixerModel.registry.get(model_type, DextraDemixerKmeansModel)()
        self.model._model_config.update(self.model_config)

    @property
    def version(self) -> str:
        """
        Return the version of the selected model implementation.
        """
        return self.model.version

    @property
    def model_type(self) -> str:
        """
        Return the registry name of the selected model implementation.
        """
        return self.model.name

    @staticmethod
    def available_methods() -> list[str]:
        """
        Return the names of registered DextraDemixer model implementations.

        Returns
        -------
        list of str
            Registry names accepted by ``model_type``.
        """
        return [k for k in ADextraDemixerModel.registry.keys()]

    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_key: str,
                              gex_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_key: str = None,
                              use_size_factor: bool | Sequence[str] | None = None,
                              outlier_threshold: float = None,
                              **kwargs) -> None:
        """
        Preprocess input counts and initialize the selected model.

        Parameters
        ----------
        mdata : mudata.MuData
            Cell-aligned pMHC count and immune-receptor modalities.
        pmhc_key : str
            Feature in ``mdata.mod[gex_key]`` to deconvolve.
        gex_key : str, default="gex"
            Key of the modality containing pMHC counts.
        neg_ctrl_key : str, optional
            Feature containing negative-control counts.
        ir_key : str, default="airr"
            Key of the immune-receptor modality.
        ir_clone_key : str, optional
            Column in ``mdata.mod[ir_key].obs`` containing clonotype IDs.
        use_size_factor : bool or sequence of str, optional
            If truthy, calculate DESeq2-style size factors. A sequence selects
            the pMHC features used; ``True`` uses every feature in ``gex_key``.
        outlier_threshold : float, optional
            Outlier threshold forwarded to the selected model implementation.
        **kwargs : Any
            Additional model-specific preprocessing and prior values.

        Raises
        ------
        KeyError
            If a requested modality, feature, or observation column is absent.
        ValueError
            If the extracted count and clonotype arrays are inconsistent.
        """
        gex = mdata.mod[gex_key]
        air = mdata.mod[ir_key]
        N = gex.shape[0]

        x = gex[:, pmhc_key].X.toarray().reshape((N,))
        x_neg = gex[:, neg_ctrl_key].X.toarray().reshape((N,)) if neg_ctrl_key else None

        c = air.obs[ir_clone_key].to_numpy().astype("int32") if ir_clone_key is not None else None

        if use_size_factor:
            pmhc_list = use_size_factor if isinstance(use_size_factor, list) else mdata[gex_key].var_names.tolist()
            x_plus = jnp.array(gex[:, pmhc_list].X.toarray(),
                               dtype=FLOAT_DTYPE)  # only used for size factor calculation
            s = self.calculate_size_factors(x_plus)
            del x_plus
        else:
            s = jnp.ones(x.shape[0], dtype=FLOAT_DTYPE)

        self._check_parameters(x, x_neg, c)
        self.model.preprocess_model_data(x=x, s=s, neg_cont=x_neg, c=c, outlier_threshold=outlier_threshold, **kwargs)

    @staticmethod
    def calculate_size_factors(counts: jax.Array) -> jax.Array:
        """
        Calculate DESeq2-style median-ratio size factors.

        Parameters
        ----------
        counts : jax.Array
            Two-dimensional count matrix with cells in rows and features in
            columns.

        Returns
        -------
        jax.Array
            One size factor per cell. Cells without a finite median ratio
            receive a factor of one.
        """

        log_counts = jnp.log(counts)
        log_counts = jnp.where(jnp.isinf(log_counts), jnp.nan, log_counts)
        log_means = jnp.nanmean(log_counts, axis=0)

        mask = jnp.isfinite(log_means) # Only use genes with non-zero geometric mean
        log_ratios = log_counts[:, mask] - log_means[mask]
        log_medians = jnp.nanmedian(log_ratios, axis=1)
        size_factors = jnp.exp(log_medians)
        size_factors = jnp.where(jnp.isnan(size_factors), 1.0, size_factors)  # Handle cells with all zero/nan counts

        return size_factors

    @staticmethod
    def get_default_sampler_config() -> dict[str, Any]:
        """
        Return the default stochastic variational inference configuration.

        Returns
        -------
        dict
            Nested configuration for optimization, tracing, and iteration.
        """

        sampler_config = {
            "svi": {
                "maxiter": 1000,
                "progress_bar": True,
                "adam": {
                    "init_value": 3e-1,
                    "transition_steps": 1,
                    "decay_rate": 0.995,
                    "end_value": 3e-3,
                },
                "tracer": {
                    "num_particles": 10,
                }
            }
        }

        return sampler_config

    def fit_svi(self, guide='normal', svi_config: dict[str, Any] | None = None,
                nof_inits: int = 100, use_minimal_loss: bool = True, rng_key: int = 998777) \
                -> az.InferenceData:
        """
        Fit the model with stochastic variational inference.

        Parameters
        ----------
        guide : {"normal", "mvnormal", "multivariatenormal"} or callable
            NumPyro autoguide name or guide constructor. A custom guide defined
            by the selected model takes precedence.
        svi_config : dict, optional
            Overrides for the optimizer, ELBO tracer, progress display, and
            maximum number of iterations.
        nof_inits : int, default=100
            Number of random initializations evaluated before optimization.
        use_minimal_loss : bool, default=True
            Use parameters from the iteration with the smallest loss instead
            of the final iteration.
        rng_key : int, default=998777
            Seed used to initialize JAX random keys.

        Returns
        -------
        arviz.InferenceData
            Posterior samples represented as ArviZ inference data.

        Raises
        ------
        Exception
            If :meth:`preprocess_model_data` has not been called.
        """

        if self.model.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.rng_key = rng_key

        if svi_config is None:
            svi_config = self.get_default_sampler_config()["svi"]

        adam_config = {**self.get_default_sampler_config()["svi"]["adam"], **svi_config.get("adam", {})}
        tracer_config = {**self.get_default_sampler_config()["svi"]["tracer"], **svi_config.get("tracer", {})}
        svi_config = {**self.get_default_sampler_config()["svi"], **svi_config}
        svi_config.pop("adam", None)
        svi_config.pop("tracer", None)

        optimizer = npy.optim.ClippedAdam(exponential_decay(**adam_config),)
        # check for custom guide in self.model otherwise use autoguide
        if guide == 'normal':
            guide = npy.infer.autoguide.AutoNormal
        elif (guide == 'mvnormal') or (guide == 'multivariatenormal'):
            guide = npy.infer.autoguide.AutoMultivariateNormal
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
        compiled_body_fn = jit(body_fn)

        with tqdm.trange(1, svi_config.get("maxiter", 1000) + 1,
                         disable=(not svi_config.get("progress_bar", False)), mininterval=10) as t:
            batch = 10
            for i in t:
                svi_state, loss, param = compiled_body_fn(svi_state, i)

                losses.append(loss)
                params.append(param)
                if i % batch == 0:
                    valid_losses = [x for x in losses[i - batch:] if x == x]
                    num_valid = len(valid_losses)
                    if num_valid == 0:
                        avg_loss = float("nan")
                    else:
                        avg_loss = sum(valid_losses) / num_valid

                    t.set_postfix_str(f"avg. loss [{i - batch + 1}-{i}]: {avg_loss:.4f}", refresh=False,)
        losses = jnp.stack(losses)
        params = params[jnp.nanargmin(losses)] if use_minimal_loss else params[-1]
        self.svi_result = SVIRunResult(params=params, losses=losses, state=svi_state)
        posterior_samples = self.guide.sample_posterior(random.PRNGKey(self.rng_key), self.svi_result.params,
                                                        sample_shape=(500,))

        # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
        posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
        inference_data = az.from_dict(posterior=posterior_samples_np)

        return inference_data

    def predict_posterior_class(self,
                                data: dict[str, Any] | None = None,
                                threshold: float = None,
                                target_fdr: float = None,
                                cred_intvl: float = None,
                                clonotype_median_p: bool = False,
                                clone_id: jax.Array | None = None,
                                ) -> tuple[jax.Array, jax.Array]:
        """
        Predict posterior binding probabilities and binary assignments.

        Parameters
        ----------
        data : dict, optional
            Preprocessed model data. The full stored data are used by default.
        threshold : float, optional
            Probability threshold in ``[0, 1]``. If neither threshold nor FDR
            control is requested, the default threshold is 0.5.
        target_fdr : float, optional
            Target Bayesian false discovery rate in ``[0, 1]``. Mutually
            exclusive with ``threshold``.
        cred_intvl : float, optional
            Required posterior probability that the FDR does not exceed
            ``target_fdr``.
        clonotype_median_p : bool, default=False
            Replace cell-level probabilities with the median probability of
            each clonotype.
        clone_id : jax.Array, optional
            Cell-aligned clonotype identifiers. Stored identifiers are used if
            available.

        Returns
        -------
        p : jax.Array
            Posterior binding probability for each cell.
        assignment : jax.Array
            Binary binding assignment for each cell.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        ValueError
            If incompatible threshold options or clonotype arguments are used.
        """
        def __return_p_summary(p_sample):
            if cred_intvl:
                p = p_sample
            else:
                p = jnp.nanmean(p_sample, axis=0)[:, 1]

            if clonotype_median_p:
                if clone_id is None:
                    raise ValueError("If `clonotype_mean_p`= True a clonotype vector `clone_id` must be specified.")
                unique_ids = np.unique(clone_id)

                if cred_intvl:
                    # mean for each clone while keeping posterior samples, shape (num_clones, num_samples, 2)
                    mean_p = np.stack([jnp.quantile(p[:, clone_id == cid], q=0.5, axis=1, method='higher') for cid in unique_ids])
                    p = mean_p[clone_id].transpose(1, 0, 2)  # shape (num_posterior_samples, num_cells, 2)

                else:
                    df = pd.DataFrame({"p": p, "clone_id": clone_id})
                    mean_p = df.groupby("clone_id")["p"].quantile(0.5, interpolation='higher')
                    p = jnp.array(mean_p.values)[clone_id]
            return p

        data = data if data is not None else self.model.data_full
        clone_id = clone_id if clone_id is not None else data.get("clone_continuous", None)
        clone_id = pd.factorize(clone_id)[0] if clone_id is not None else None
        
        if self.sampler is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        # posterior probability of belonging to the binding class
        predictive = npy.infer.Predictive(self.model.model, guide=self.guide, params=self.svi_result.params,
                                            num_samples=500)
        samples = predictive(jax.random.PRNGKey(self.rng_key), data=data)  # self.rng_key
        p = __return_p_summary(jnp.exp(samples["log_p"]))

        if cred_intvl is not None:
            p, assignment, threshold = self._predict_posterior_class_dist(p, target_fdr, cred_intvl)
        else:
            assignment = self._predict_posterior_class(p, threshold, target_fdr)

        return p, assignment

    def summary(self) -> pd.DataFrame:
        """
        Summarize fitted model parameters other than class probabilities.

        Returns
        -------
        pandas.DataFrame
            ArviZ posterior summary statistics.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if self.trace is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

        posterior_samples = self.guide.sample_posterior(random.PRNGKey(self.rng_key), self.svi_result.params,
                                                        sample_shape=(500,))

        # Convert posterior_samples from JAX arrays to NumPy arrays and reshape
        posterior_samples_np = {k: np.array(v)[np.newaxis, ...] for k, v in posterior_samples.items()}
        inference_data = az.from_dict(posterior=posterior_samples_np)
        return az.summary(inference_data, var_names=["~log_p"])

    def __make_arvis(self):
        self.trace = az.from_numpyro(self.sampler)
        return self.trace

    @staticmethod
    def _predict_posterior_class_dist(p_samples, target_fdr, cred_intvl, nof_thresh=100):
        r"""
        Posterior BFDR thresholding (Newton et al. 2004, extended with posterior uncertainty).

        Given posterior draws of signal probabilities \(p_i^{(s)}\), this method computes
        the posterior distribution of the global FDR across candidate thresholds \(\tau\).
        For each \(\tau\), the per-draw FDR is

        \[
        \text{FDR}^{(s)}(\tau) =
        \frac{\sum_i (1 - p_i^{(s)}) \mathbf{1}[p_i^{(s)} \geq \tau]}
             {\sum_i \mathbf{1}[p_i^{(s)} \geq \tau]} .
        \]

        The selected threshold is the largest \(\tau\) such that
        \(\Pr(\text{FDR}(\tau) \leq \alpha \mid \text{data}) \geq \text{cred\_level}\).
        This provides a conservative extension of the DPP rule that accounts for
        posterior uncertainty in posterior class probabilities.

        Parameters
        ----------
        p_samples : jax.Array
            Posterior samples with shape ``(n_draws, n_samples, 2)``.
        target_fdr : float
            Target false discovery rate :math:`\alpha \in [0, 1]`.
        cred_intvl : float
            Required posterior probability of FDR control.
        nof_thresh : int, default=100
            Number of candidate thresholds.

        Returns
        -------
        p_mean : jax.Array
            Mean posterior binding probabilities.
        assignment : jax.Array
            Binary binding assignments.
        threshold : float
            Selected probability threshold :math:`\tau`.
        """
        p_samples = p_samples[:, :, 1]
        p_mean = jnp.mean(p_samples, axis=0)
        lfdr = 1.0 - p_samples
        candidate_thresh = jnp.linspace(0.0, 1.0, nof_thresh + 2)[1:-1]

        def eval_threshold(_, tau):
            disc = p_samples >= tau
            n_disc = disc.sum(axis=1)
            sum_lfdr = jnp.sum(jnp.where(disc, lfdr, 0.0), axis=1)
            gfdr = jnp.where(n_disc > 0, sum_lfdr / n_disc, 0.0)
            valid = jnp.mean(gfdr <= target_fdr) >= cred_intvl
            mean_n_disc = jnp.mean(n_disc)
            return None, (valid, mean_n_disc)

        _, (valid_thr, n_discoveries) = jax.lax.scan(eval_threshold, None, candidate_thresh)

        threshold_idx = jnp.argmax(jnp.where(valid_thr, n_discoveries, -1.0))
        threshold = jnp.where(jnp.any(valid_thr), candidate_thresh[threshold_idx], 1.0)

        assignment = (p_mean >= threshold).astype(jnp.int32)
        return p_mean, assignment, threshold

    def get_posterior_samples(self, num_samples: int = 1000, seed: int = 42) -> dict[str, jax.Array]:
        """
        Return posterior means and derived negative-binomial parameters.

        Parameters
        ----------
        num_samples : int, default=1000
            Number of posterior draws used to calculate the summaries.
        seed : int, default=42
            Seed used to initialize the NumPyro random key.

        Returns
        -------
        dict of str to jax.Array
            Posterior means and derived quantities such as ``q``, ``w``,
            ``alpha``, and ``overdispersion``. Negative-control quantities are
            included when that model component is present.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if self.trace is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

        predictive = npy.infer.Predictive(self.guide, params=self.svi_result.params, num_samples=num_samples)
        posterior_samples = predictive(jax.random.PRNGKey(seed), data=None)

        # Extract mean from posterior samples
        q = posterior_samples["delta_q"].mean(0).cumsum(0)
        w = posterior_samples["w"].mean(0)

        if w.ndim > 2:
            # w is per clone, transform to per cell and take mean over all cells
            w_cell = w[self.model.data["clone_continuous"]]
            w_mean_over_cells = w_cell.mean(0)
        else:
            w_mean_over_cells = w

        overdispersion = posterior_samples["overdispersion"].mean(0) + 1
        # alpha.shape = (2, )
        alpha = q ** 2 / (q * (overdispersion) - q)
        if self.model._model_config['alpha_offset']:
            alpha = alpha + jnp.array([0, self.model._model_config['alpha_offset']])

        alpha_mean_over_cells = alpha

        posterior_samples_mean = {"q": q, "w": w, "alpha": alpha,
                                  "w_mean_over_cells": w_mean_over_cells, "alpha_mean_over_cells": alpha_mean_over_cells}
        posterior_samples_mean["overdispersion"] = overdispersion
        # Negative control model
        if 'noise_mean_inv_inc' in posterior_samples:
            s = jnp.ones(self.model.data["x"].shape[0]) if self.model.data["s"] is None else self.model.data["s"]

            posterior_samples_mean['noise_mean_inv_inc'] = posterior_samples['noise_mean_inv_inc'].mean(0)
            posterior_samples_mean['noise_overdisp_inv_inc'] = posterior_samples['noise_overdisp_inv_inc'].mean(0)

            q_neg = jnp.clip(s * q[0] / posterior_samples_mean['noise_mean_inv_inc'], a_min=1e-3)
            overdispersion_neg = jnp.clip(
                posterior_samples_mean['overdispersion'][0] / posterior_samples_mean['noise_overdisp_inv_inc'],
                a_min=1.0 + 1e-3)
            alpha_neg = q_neg ** 2 / (q_neg * overdispersion_neg - q_neg)
            posterior_samples_mean['q_neg'] = q_neg.mean()
            posterior_samples_mean['alpha_neg'] = alpha_neg.mean()
            posterior_samples_mean['overdispersion_neg'] = overdispersion_neg
        return posterior_samples_mean

    def plot_results(self, assignment, p_pred, y_true=None, seed=42, show=False, return_plt=False, data=None) -> None:
        """
        Plot assignments, probabilities, and fitted mixture components.

        Parameters
        ----------
        assignment : array-like
            Predicted binary assignments.
        p_pred : array-like
            Posterior binding probabilities.
        y_true : array-like, optional
            True assignments. Zeros are used when labels are unavailable.
        seed : int, default=42
            Seed used when drawing posterior parameter summaries.
        show : bool, default=False
            Display the figure with :func:`matplotlib.pyplot.show`.
        return_plt : bool, default=False
            Leave the current Matplotlib figure open instead of closing it.
        data : dict, optional
            Preprocessed model data to plot. Stored full data are used by
            default.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """

        if self.trace is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

        if y_true is None:
            # Create pseudo ground-truth label for plotting
            y_true = np.zeros_like(assignment)
        
        data = data if data is not None else self.model.data_full
        
        plt.figure(figsize=(10, 7))

        # FIRST COLUMN - TRUE CLASS ASSIGNMENT
        # Plot data colored in TRUE class assignment
        plt.subplot(3, 3, 1)
        ax = sns.histplot(x=data["x"], hue=y_true, discrete=True, element="step", alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("True class")
        leg.set_frame_on(False)
        sns.despine()
        plt.title("True class assignment")

        # Plot data colored in TRUE class assignment log-scale
        plt.subplot(3, 3, 4)
        ax = sns.histplot(x=data["x"], hue=y_true, discrete=True, element="step", alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("True class")
        leg.set_frame_on(False)
        sns.despine()
        plt.yscale("log")
        plt.title("True class assignment log-scale")

        # Plot UMI count vs predicted probability colored in TRUE class assignment
        plt.subplot(3, 3, 7)
        ax = sns.scatterplot(x=data["x"], y=p_pred, hue=y_true, alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("True class")
        leg.set_frame_on(False)
        sns.despine()
        plt.xlabel("UMI count")
        plt.ylabel("Posterior probability")
        plt.title("Pred prob and true label")

        # SECOND COLUMN - PREDICTED CLASS ASSIGNMENT
        # Plot data colored in PREDICTED class assignment
        plt.subplot(3, 3, 2)
        ax = sns.histplot(x=data["x"], hue=assignment, discrete=True, element="step", alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("Pred class")
        leg.set_frame_on(False)
        sns.despine()
        plt.title("Predicted class assignment")

        # Plot data colored in PREDICTED class assignment in log scale
        plt.subplot(3, 3, 5)
        ax = sns.histplot(x=data["x"], hue=assignment, discrete=True, element="step", alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("Pred class")
        leg.set_frame_on(False)
        sns.despine()
        plt.yscale("log")
        plt.title("Predicted class assignment log-scale")

        # Plot UMI count vs predicted probability colored in PREDICTED class assignment
        plt.subplot(3, 3, 8)
        ax = sns.scatterplot(x=data["x"], y=p_pred, hue=assignment, markers={0: ".", 1: "X"}, alpha=0.3)
        leg = ax.get_legend()
        leg.set_title("Pred class")
        leg.set_frame_on(False)
        sns.despine()
        plt.xlabel("UMI count")
        plt.ylabel("Posterior probability")
        plt.title("Pred prob and pred label")

        # THIRD COLUMN - POSTERIOR DISTRIBUTION OF NEGATIVE BINOMIAL
        # Plot posterior distribution of Negative Binomial
        posterior_samples = self.get_posterior_samples(num_samples=1000, seed=seed)
        q = posterior_samples["q"]
        w = posterior_samples["w"]
        alpha = posterior_samples["alpha"]
        x = np.arange(0, data["x"].max())

        prob0 = jnp.exp(npd.NegativeBinomial2(q[0], alpha[0]).log_prob(x))
        prob1 = jnp.exp(npd.NegativeBinomial2(q[1], alpha[1]).log_prob(x))

        # Individual Negative Binomial
        plt.subplot(3, 3, 6)
        ax1 = sns.lineplot(x=np.arange(0, data["x"].max()), y=prob0,
                            label=fr"$q={q[0]:.1f}\ \alpha={alpha[0]:.1f}$", color=sns.color_palette('tab10')[0])
        ax2 = ax1.twinx()
        sns.lineplot(x=np.arange(0, data["x"].max()), y=prob1, ax=ax2,
                        label=fr"$q={q[1]:.1f}\ \alpha={alpha[1]:.1f}$", color=sns.color_palette('tab10')[1])
        handles = ax1.lines + ax2.lines
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, frameon=False, loc='best')
        ax2.get_legend().remove()
        sns.despine()
        plt.title("Posterior NB components")
        plt.ylabel("Probability")

        # Mixture model
        plt.subplot(3, 3, 3)
        # w.shape = (2,)
        prob0_mix = prob0 * w[0]
        prob1_mix = prob1 * w[1]
        w_mean = w

        sns.lineplot(x=x, y=prob0_mix.reshape(-1),
                        label=fr"$q={q[0]:.1f}\ \alpha={alpha[0]:.1f}$", linewidth=3)
        sns.lineplot(x=x, y=prob1_mix.reshape(-1),
                    label=fr"$q={q[1]:.1f}\ \alpha={alpha[1]:.1f}$", linewidth=3)
        sns.lineplot(x=x, y=(prob0_mix + prob1_mix).reshape(-1), linewidth=3, color="k",
                        label=f"w={w_mean[0]:.2f}, {w_mean[1]:.3f}", linestyle="--")
        plt.legend(frameon=False)
        sns.despine()
        plt.title("Posterior Mixture NB")
        plt.ylabel("Probability")

        plt.tight_layout()

        if show:
            plt.show()
        if return_plt:
            return
        plt.close()

    def save_model(self, filepath: str | os.PathLike) -> None:
        """
        Save model state to a pickle file.

        Parameters
        ----------
        filepath : str or os.PathLike
            Destination checkpoint path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(vars(self), f)

    def load_model(self, filepath: str | os.PathLike) -> Self:
        """
        Load checkpoint state into this instance.

        Parameters
        ----------
        filepath : str or os.PathLike
            Pickle checkpoint path.

        Returns
        -------
        DextraDemixer
            This instance after its state has been replaced.

        Notes
        -----
        Pickle files can execute arbitrary code. Only load trusted checkpoints.
        """
        with open(filepath, 'rb') as f:
            ckpt = pickle.load(f)
        self.__dict__.update(ckpt)
        return self

    @classmethod
    def from_ckpt(cls, filepath: str | os.PathLike) -> Self:
        """
        Construct an instance directly from a checkpoint.

        Parameters
        ----------
        filepath : str or os.PathLike
            Pickle checkpoint path.

        Returns
        -------
        DextraDemixer
            New instance populated without calling ``__init__``.

        Notes
        -----
        Pickle files can execute arbitrary code. Only load trusted checkpoints.
        """
        with open(filepath, 'rb') as f:
            ckpt = pickle.load(f)
        self = cls.__new__(cls)
        self.__dict__.update(ckpt)
        return self


class ADextraDemixerModel(metaclass=RegisteredModel):
    """
    Define the extension contract for registered DextraDemixer models.

    Attributes
    ----------
    data : dict or None
        Preprocessed data used for model fitting.
    name : str
        Registry name of the model implementation.
    version : str
        Version of the model implementation.
    """

    def __init__(self):
        self._name = "Abstract"
        self._version = "0.0.0"
        self._data = None
        self._kmeans_dict = None

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, jax.Array],
                              s: Union[pd.Series, np.ndarray, jax.Array] = None,
                              neg_cont: Union[pd.Series, np.ndarray, jax.Array] = None,
                              c: Union[pd.Series, np.ndarray, jax.Array] = None,
                              **kwargs) -> None:
        """
        Convert input arrays to the representation used by NumPyro models.

        Parameters
        ----------
        x : pandas.Series, numpy.ndarray, or jax.Array
            Target pMHC counts.
        s : pandas.Series, numpy.ndarray, or jax.Array, optional
            Cell-level size factors.
        neg_cont : pandas.Series, numpy.ndarray, or jax.Array, optional
            Negative-control counts.
        c : pandas.Series, numpy.ndarray, or jax.Array, optional
            Cell-level clonotype identifiers.
        **kwargs : Any
            Model-specific preprocessing values.
        """
        clone = None if c is None else jnp.array(c, dtype=INT_DTYPE)
        zscore = jnp.abs((x - jnp.mean(x)) / jnp.std(x))
        outlier_threshold = 100 # TODO Hardcoded
        # With outliers
        self.data_full = {"x": jnp.array(x, dtype=INT_DTYPE),
                          "s": None if s is None else jnp.array(s, dtype=FLOAT_DTYPE),
                          "x_neg": None if neg_cont is None else jnp.array(neg_cont, dtype=FLOAT_DTYPE),
                          "clone": clone,
                          # If clone is not contiuous, then there will be problems with indexing
                          "clone_continuous": None if clone is None else jnp.searchsorted(jnp.unique(clone), clone),
                          }
        # Without outliers
        self.data = {"x": jnp.array(x[jnp.where(zscore < outlier_threshold)], dtype=INT_DTYPE),
                     "s": jnp.array(s[jnp.where(zscore < outlier_threshold)], dtype=FLOAT_DTYPE) if s is not None else None,
                     "x_neg": jnp.array(neg_cont[jnp.where(zscore < outlier_threshold)], dtype=FLOAT_DTYPE) if neg_cont is not None else None,
                     "clone": jnp.array(clone[jnp.where(zscore < outlier_threshold)], dtype=INT_DTYPE) if clone is not None else None,
                     "clone_continuous": None if clone is None else jnp.searchsorted(jnp.unique(clone), clone[jnp.where(zscore < outlier_threshold)]),
                     }

    def _init_kmeans(self, scale_factor=1.0, outlier_threshold=None) -> Dict:
        """
        Initialize two K-means clusters and estimate prior parameters.

        Parameters
        ----------
        scale_factor : float, default=1.0
            Reserved scaling value for model-specific initialization.
        outlier_threshold : float, optional
            Reserved outlier threshold for model-specific initialization.

        Returns
        -------
        dict
            Cluster labels, means, variances, proportions, and Dirichlet
            concentration parameters.
        """
        x = self.data["x"].copy()
        x_no_outliers = x
        clone = self.data.get("clone_continuous", None)
        n_clusters = 2  # KMeans with 2 clusters

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, init=np.vstack([np.min(x_no_outliers), np.max(x_no_outliers)]), n_init="auto").fit(x_no_outliers.reshape(-1, 1))
        labels = kmeans.predict(x.reshape(-1, 1))

        if labels.sum() <= 3:
            # Assign highest three values to cluster 1 and the rest to cluster 0
            sorted_indices = np.argsort(x)
            labels[sorted_indices[-3:]] = 1
        
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
        """
        Define the NumPyro probabilistic model.

        Parameters
        ----------
        **kwargs : Any
            Model-specific data and configuration overrides.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_default_model_config(self) -> Dict:
        """
        Return the model's default configuration.

        Returns
        -------
        dict
            Model prior and initialization values.
        """
        return {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Return the model's registry name.
        """
        return self._name

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """
        Return the model implementation version.
        """
        return self._version

    @property
    def data(self) -> Dict:
        """
        Return the preprocessed fitting data.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class DextraDemixerKmeansModel(ADextraDemixerModel):
    """
    Parameterize mixture-model priors with a two-cluster K-means fit.

    Notes
    -----
    This implementation estimates fixed prior parameters from the observed
    clusters instead of learning them from hyperpriors.
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
            "overdispersion_scale_prior": 1e-2,
            "alpha_offset": 0.0,
        }

    @property
    def name(self) -> str:
        """
        Return the model's registry name.
        """
        return self._name

    @property
    def version(self) -> str:
        """
        Return the model implementation version.
        """
        return self._version

    def preprocess_model_data(self,
                              x: Union[pd.Series, np.ndarray, jax.Array],
                              s: Union[pd.Series, np.ndarray, jax.Array] = None,
                              neg_cont: Union[pd.Series, np.ndarray, jax.Array] = None,
                              c: Union[pd.Series, np.ndarray, jax.Array] = None,
                              scale_factor: float = 1.0,
                              outlier_threshold: float = None,
                              **kwargs) -> None:
        """
        Preprocess counts and estimate K-means initialization values.

        Parameters
        ----------
        x : pandas.Series, numpy.ndarray, or jax.Array
            Target pMHC counts.
        s : pandas.Series, numpy.ndarray, or jax.Array, optional
            Cell-level size factors.
        neg_cont : pandas.Series, numpy.ndarray, or jax.Array, optional
            Negative-control counts.
        c : pandas.Series, numpy.ndarray, or jax.Array, optional
            Cell-level clonotype identifiers.
        scale_factor : float, default=1.0
            Scaling value used during prior initialization.
        outlier_threshold : float, optional
            Threshold used during K-means initialization.
        **kwargs : Any
            Additional model-specific preprocessing values.
        """

        super().preprocess_model_data(x=x, s=s, neg_cont=neg_cont, c=c, **kwargs)
        self._kmeans_dict = self._init_kmeans(scale_factor=scale_factor,
                                              outlier_threshold=outlier_threshold)
        self._model_config.update(self._kmeans_dict)

    def get_default_model_config(self) -> Dict:
        """
        Return default and data-derived model configuration values.

        Returns
        -------
        dict
            Prior parameters and K-means initialization results.
        """
        return self._model_config

    def model(self, data=None, **kwargs):
        """
        Define the probabilistic model based on the preprocessed data and KMeans initialization.

        Parameters
        ----------
        data : dict, optional
            Preprocessed data. Stored fitting data are used by default.
        **kwargs : Any
            Model configuration overrides.

        Raises
        ------
        RuntimeError
            If no preprocessed data are available.
        """

        model_config = {**self.get_default_model_config(), **kwargs.get("model_config", {})}
        if data is None:
            if self.data is None:
                raise RuntimeError("Model was not properly initialized. Please call `preprocess_model_data` first.")
            data = self.data

        x = data["x"]
        s = data["s"]
        x_neg = data["x_neg"]
        N_sample = x.shape[0]
        K = 2

        # Extract hyperpriors
        cluster_means = model_config["cluster_means"]
        cluster_variances = model_config["cluster_variances"]
        tau_concentration_prior = model_config["tau_concentration_prior"]
        overdispersion_scale_prior = model_config["overdispersion_scale_prior"]
        alpha_offset = model_config.get("alpha_offset", False)

        # Cluster probability prior
        w = npy.sample("w", npd.Dirichlet(tau_concentration_prior))
        z = npd.Categorical(probs=w)

        # Convert kmeans priors to deltas, due to cumsum ordering
        mean_deltas = jnp.array([max(cluster_means[0], 1e-1), cluster_means[1] - cluster_means[0]])
        var_deltas = jnp.array([max(cluster_variances[0], 1e-1), max(cluster_variances[1] - cluster_variances[0], 1)])

        # Convert kmeans parameters to lognormal parameters with target mean and variance
        # NB mean parameter: q_prior ~ LogNormal(mu_q, sigma_q), with cluster means and variances
        sigma2_q_prior = jnp.log(var_deltas / mean_deltas ** 2 + 1)
        sigma_q_prior = jnp.maximum(jnp.sqrt(sigma2_q_prior), 0.01)  # avoid sigma=0
        mu_q_prior = jnp.log(mean_deltas) - sigma2_q_prior / 2

        # Sample delta_q from lognormal distribution and cumsum to create ordered q
        with npy.plate("cluster_axis", K):
            delta_q = npy.sample("delta_q", npd.LogNormal(loc=mu_q_prior, scale=sigma_q_prior))
        q = npy.deterministic("q", jnp.cumsum(delta_q, axis=0))

        # NB concentration parameter: alpha = q^2 / (q * overdispersion - q), overdispersion ~ HalfCauchy(1) + 1
        overdispersion_prior_dist = npd.HalfCauchy(overdispersion_scale_prior)
        # For each mixture component, we have one alpha parameter
        with npy.plate("cluster_axis", K):
            overdispersion = npy.sample("overdispersion", overdispersion_prior_dist) + 1
        # Make sure that alpha > 1 to prevent exponential dist for the second component
        if alpha_offset:
            alpha = npy.deterministic("alpha", q**2 / (q * overdispersion - q) + jnp.array([0, alpha_offset]))
        else:
            alpha = npy.deterministic("alpha", q**2 / (q * overdispersion - q))

        if x_neg is not None:
            s_q = npy.sample("s_q", npd.LogNormal(0.9692917285815055, 0.6293977074906485))
            q_neg = npy.deterministic("q_neg", jnp.clip(s * q[0] / s_q, a_min=1e-3))
            s_alpha = npy.sample("s_alpha", npd.LogNormal(0.19724303327974974, 0.43970806321879075))
            overdispersion_neg = npy.deterministic("overdispersion_neg", jnp.clip(overdispersion[0] / s_alpha, a_min=1.0 + 1e-3))
            with npy.plate("sample_axis", N_sample):
                alpha_neg = npy.deterministic("alpha_neg", q_neg ** 2 / (q_neg * overdispersion_neg - q_neg))
                yhat_neg = npy.sample("yhat_neg", obs=x_neg,
                                        fn=npd.NegativeBinomial2(mean=q_neg, concentration=alpha_neg, ))

        # Sample from the mixture model
        with npy.plate("sample_axis", N_sample):
            # target pMHC
            mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=s[:,None]*q, concentration=alpha))

            yhat = npy.sample("yhat", mixture, obs=x)

            # Membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))
