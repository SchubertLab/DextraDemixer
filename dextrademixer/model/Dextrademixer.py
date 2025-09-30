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
                raise ValueError("If `mode`= C a clonotype vector `ir_clone_key` must be specified.")

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

    def fit_svi(self, guide=npy.infer.autoguide.AutoMultivariateNormal, svi_config: Dict[str, Union[int, float]] = None,
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
                                target_fdr: float = None,
                                quantile: float = None,
                                cred_intvl: float = None,
                                clonotype_adherence: bool = False
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
            quantile: (Optional) whether and what lower quantile should be used instead of the population mean as conservative
                      measure of p. quantile should be in (0, 0.5]
            cred_intvl: (Optional) instead of using the summarized class probability we estimate a distribution
                        over Pr(FDR(t)≤alpha|posterior)≥cred_intvl
            clonotype_adherence: (Optional) instead of using posterior class assignment per cell use clonotype
                                 probability vector if available.
        Returns:
            A tuple (p, assignment) of arrays with p being the posterior probability of binding and assignment the
            class assignment decision
        """
        def __return_p_summary(p_sample):
            if quantile:
                return jnp.quantile(p_sample, quantile, axis=0)[:, 1]
            elif cred_intvl:
                return p_sample
            else:
                return jnp.nanmean(p_sample, axis=0)[:, 1]

        if self.sampler is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        # posterior probability of belonging to the binding class
        if self.is_svi:
            if clonotype_adherence and self.model.data["clone"] is not None:
                #TODO ALTERNATIVE USE MAJORITY VOTING IN CLONE?
                posterior_samples = self.guide.sample_posterior(random.PRNGKey(self.rng_key), self.svi_result.params,
                                                                sample_shape=(500,))
                p = __return_p_summary(posterior_samples["w"])

            else:
                predictive = npy.infer.Predictive(self.model.model, guide=self.guide, params=self.svi_result.params,
                                                  num_samples=500)
                samples = predictive(jax.random.PRNGKey(self.rng_key))  # self.rng_key
                p = __return_p_summary(jnp.exp(samples["log_p"]))

        else:
            if clonotype_adherence and self.model.data["clone"] is not None:
                p = __return_p_summary(self.sampler.get_samples()["w"])
            else:
                p = __return_p_summary(jnp.exp(self.sampler.get_samples()["log_p"][..., [0,1]]))

        if cred_intvl is not None:
            p, assignment, threshold = self._predict_posterior_class_dist(p, target_fdr, cred_intvl)
        else:
            assignment = self._predict_posterior_class(p, threshold, target_fdr)

        if clonotype_adherence and self.model.data["clone"] is not None:
            assignment = assignment[self.model.data["clone"]]
            p = p[self.model.data["clone"]]

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

    @staticmethod
    def _predict_posterior_class_dist(p_samples,  target_fdr, cred_intvl, nof_thresh=100):
        """
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

        Args:
            p_samples (Array): Posterior samples of signal probabilities,
                shape (n_draws, n_samples).
            target_fdr (float): Target false discovery rate \(\alpha \in [0,1]\).
            cred_intvl (float, optional): Credibility requirement for
                FDR control, \(cred_intvl \in [0.5,1)\)

        returns:
            Tuple[Array, Array, float]:
                - Posterior mean posterior class probabilities (\( \hat{p}_i \)), shape (n_samples,)
                - Hard assignments (0/1), shape (n_samples,)
                - Selected threshold \(\tau\)
        """
        p_samples = p_samples[:,:, 1]
        p_mean = jnp.mean(p_samples, axis=0) # (n, )

        # Candidate thresholds = unique posterior means
        # or even grid in (0, 1)
        #candidate_thresh = jnp.sort(jnp.unique(p_mean))  # (T,)
        candidate_thresh = jnp.linspace(0.0, 1.0, nof_thresh+2)[1:-1]

        # Expand shapes: (n_draws, n) vs (T,)
        # -> mask has shape (T, n_draws, n)
        mask = p_samples[None, :, :] >= candidate_thresh[:, None, None]

        # number of discoveries per draw: (T, n_draws)
        n_disc = mask.sum(axis=2)

        # local fdr = 1 - p
        lfdr = 1.0 - p_samples  # (n_draws, n)

        # numerator = sum of lfdr among discoveries: (T, n_draws)
        num = jnp.einsum("tns,ns->tn", mask, lfdr)

        # avoid div-by-zero: set fdr_draws=0 where n_disc=0
        fdr_draws = jnp.where(n_disc > 0, num / n_disc, 0.0)

        # posterior probability that FDR ≤ target: (T,)
        # keep thresholds that pass cred_level
        valid = (fdr_draws <= target_fdr).mean(axis=1) >= cred_intvl

        if valid.any():
            n_discoveries = n_disc.mean(axis=1)
            # select threshold that passes credibility interval on FDR with the largest number of discoveries
            threshold_idx = jnp.argmax(jnp.where(valid, n_discoveries, -1))
            threshold = candidate_thresh[threshold_idx]
        else:
            threshold = 1.0

        assignment = (p_mean >= threshold).astype(jnp.int32)
        return p_mean, assignment, threshold

    def plot_results(self, assignment, p_pred, y_true=None, seed=42, config=''):

        if self.trace is None and self.svi_result is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

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
        sns.scatterplot(x=self.model.data["x"], y=p_pred, hue=y_true)
        sns.despine()
        plt.ylabel("Posterior probability")
        plt.title("Predicted probability of UMI count with true label")

        # Plot posterior distribution of Negative Binomial
        if self.is_svi: # svi
            predictive = npy.infer.Predictive(self.guide, params=self.svi_result.params, num_samples=1000)
            posterior_samples = predictive(jax.random.PRNGKey(seed), data=None)
        else: # mcmc inference
            posterior_samples = self.sampler.get_samples()

        # Extract mean from posterior samples
        q = posterior_samples["q"].mean(0)
        w = posterior_samples["w"].mean(0)
        x = np.arange(0, self.model.data["x"].max())

        # extract alpha, which depends on the mode and alpha_model
        if self.alpha_model == 'kmeans':
            # shape will be defined by mode, mode=='C': (C, 1), mode=='I': (1, 2)
            alpha = posterior_samples["alpha"].mean(0)
        else:
            overdispersion = posterior_samples["overdispersion"].mean(0) + 1
            if self.mode == "C":
                # alpha.shape = (C, 1)
                q_weighted = (w * q).mean(1)
                alpha = q_weighted**2 / (q_weighted * (overdispersion) - q_weighted)
            elif self.mode == "I":
                # alpha.shape = (1, 2)
                alpha = q**2 / (q * (overdispersion) - q)

        if self.mode == "C":
            # alpha_weighted is the mean of alpha weighted by w for each cell with shape (2, )
            # This reflects better the contribution of each alpha on average for the binder and non-binder NB component
            alpha_weighted = (w[self.model.data["clone_continuous"]] * alpha[self.model.data["clone_continuous"]][:, None])
            alpha_weighted = alpha_weighted.mean(0) / w.mean(0)

            # pdf for each cell
            prob0 = jnp.exp(npd.NegativeBinomial2(mean=q[0], concentration=alpha[self.model.data["clone_continuous"]][:, np.newaxis]).log_prob(x))
            prob1 = jnp.exp(npd.NegativeBinomial2(mean=q[1], concentration=alpha[self.model.data["clone_continuous"]][:, np.newaxis]).log_prob(x))

            # Individual Negative Binomial
            plt.subplot(3, 4, 8)
            ax1 = sns.lineplot(x=x, y=prob0.mean(0), c=sns.color_palette('tab10')[0],
							   label=f"q={q[0]:.2f} alpha={alpha_weighted[0]:.2f}")
            plt.fill_between(x, np.quantile(prob0, 0.05, axis=0), np.quantile(prob0, 0.95, axis=0), alpha=0.3,
							 label='5%-95% percentile')
            ax2 = ax1.twinx()
            sns.lineplot(x=x, y=prob1.mean(0), ax=ax2, c=sns.color_palette('tab10')[1],
						 label=f"q={q[1]:.2f} alpha={alpha_weighted[1]:.2f}")
            plt.fill_between(x, np.quantile(prob1, 0.05, axis=0), np.quantile(prob1, 0.95, axis=0), alpha=0.3,
							 label='5%-95% percentile')
            plt.legend()
            sns.despine()
            plt.title("Posterior NB without mixing weights")
            plt.ylabel("Probability")

            # Mixture model
            # Use different w for each clonotype and hence cell: w.shape = (#clonotypes, 2)
            prob0_mix = prob0 * w[self.model.data["clone_continuous"]][:, 0:1]
            prob1_mix = prob1 * w[self.model.data["clone_continuous"]][:, 1:2]
            w_mean = w[self.model.data["clone_continuous"]].mean(0)  # mean over all clonotypes

            plt.subplot(3, 4, 4)
            sns.lineplot(x=x, y=prob0_mix.mean(0), c=sns.color_palette('tab10')[0],
						 label=f"q={q[0]:.2f} alpha={alpha_weighted[0]:.2f}", color=sns.color_palette('tab10')[0])
            sns.lineplot(x=x, y=prob1_mix.mean(0), c=sns.color_palette('tab10')[1],
						 label=f"q={q[1]:.2f} alpha={alpha_weighted[1]:.2f}", color=sns.color_palette('tab10')[0])
            sns.lineplot(x=x, y=prob0_mix.mean(0)+prob1_mix.mean(0), c="k", linestyle=":",
						 label=f"mixture w={w_mean[0]:.4f}, {w_mean[1]:.4f}")
            plt.fill_between(x, np.quantile(prob0_mix, 0.05, axis=0), np.quantile(prob0_mix, 0.95, axis=0),
							 alpha=0.3, label='5%-95% percentile')
            plt.fill_between(x, np.quantile(prob1_mix, 0.05, axis=0), np.quantile(prob1_mix, 0.95, axis=0),
							 alpha=0.3, label='5%-95% percentile')
            plt.legend()
            sns.despine()
            plt.title("Posterior Mixture NB")
            plt.ylabel("Probability")

        elif self.mode == "I":
            npd.NegativeBinomial2(mean=q, concentration=alpha)
            prob0 = jnp.exp(npd.NegativeBinomial2(q[0], alpha[0]).log_prob(x))
            prob1 = jnp.exp(npd.NegativeBinomial2(q[1], alpha[1]).log_prob(x))

            # Individual Negative Binomial
            plt.subplot(3, 4, 8)
            ax1 = sns.lineplot(x=np.arange(0, self.model.data["x"].max()), y=prob0,
							   label=f"q={q[0]:.2f} alpha={alpha[0]:.2f}", color=sns.color_palette('tab10')[0])
            ax2 = ax1.twinx()
            sns.lineplot(x=np.arange(0, self.model.data["x"].max()), y=prob1, ax=ax2,
						 label=f"q={q[1]:.2f} alpha={alpha[1]:.2f}", color=sns.color_palette('tab10')[1])
            sns.despine()
            plt.title("Posterior NB without mixing weights")
            plt.ylabel("Probability")

            # Mixture model
            plt.subplot(3, 4, 4)
            if self.model.data["clone_continuous"] is not None:
                # Use different w for each clonotype: w.shape = (#clonotypes, 2)
                # probX = (max UMI)
                prob0_mix = prob0[None,] * w[:, 0:1]
                prob1_mix = prob1[None,] * w[:, 1:2]
                w_mean = w.mean(0)
                sns.lineplot(x=x, y=prob0_mix.mean(0), c=sns.color_palette('tab10')[0],
                             label=f"q={q[0]:.2f} alpha={alpha[0]:.2f}")
                sns.lineplot(x=x, y=prob1_mix.mean(0), c=sns.color_palette('tab10')[1],
                             label=f"q={q[1]:.2f} alpha={alpha[1]:.2f}")
                sns.lineplot(x=x, y=prob0_mix.mean(0) + prob1_mix.mean(0), c="k", linestyle=":",
                             label=f"mixture w={w_mean[0]:.4f}, {w_mean[1]:.4f}")
                plt.fill_between(x, np.quantile(prob0_mix, 0.05, axis=0), np.quantile(prob0_mix, 0.95, axis=0),
                                 alpha=0.3, label='5%-95% percentile')
                plt.fill_between(x, np.quantile(prob1_mix, 0.05, axis=0), np.quantile(prob1_mix, 0.95, axis=0),
                                 alpha=0.3, label='5%-95% percentile')
                plt.legend()
            else:
                # w.shape = (2,)
                prob0_mix = prob0 * w[0]
                prob1_mix = prob1 * w[1]
                w_mean = w

                sns.lineplot(x=x, y=prob0_mix.reshape(-1),
                             label=f"q={q[0]:.2f} alpha={alpha[0]:.2f}", linewidth=3)
                sns.lineplot(x=x, y=prob1_mix.reshape(-1),
                            label=f"q={q[1]:.2f} alpha={alpha[1]:.2f}", linewidth=3)
                sns.lineplot(x=x, y=(prob0_mix + prob1_mix).reshape(-1), linewidth=3, color="k",
                             label=f"mixture w={w_mean[0]:.4f}, {w_mean[1]:.4f}", linestyle="--")
            sns.despine()
            plt.title("Posterior Mixture NB")
            plt.ylabel("Probability")

        # Plot kmeans clusters
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
        try:
            f1 = f1_score(assignment, y_true)
        except:
            # if y_true is None or str
            f1 = -1
        plt.suptitle(config.replace("_", " ")
					 .replace("ncell", "\nncell") + f"\nF1-score {f1:.3f}",)
        os.makedirs("figs", exist_ok=True)
        plt.savefig(f"figs/{config}.png")
        plt.show()
        plt.close()

        return q, alpha, w


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
                              s: Union[pd.Series, np.ndarray, Array] = None,
                              neg_cont: Union[pd.Series, np.ndarray, Array] = None,
                              c: Union[pd.Series, np.ndarray, Array] = None,
                              sigma: Union[np.ndarray, Array] = None,
                              mode: str = "H",
                              alpha_model="overdispersion",
                              **kwargs):
        """
        """
        clone = None if c is None else jnp.array(c, dtype=INT_DTYPE)
        self.data = {"x": jnp.array(x, dtype=INT_DTYPE),
                     "s": None if s is None else jnp.array(s, dtype=FLOAT_DTYPE),
                     "x_neg": None if neg_cont is None else jnp.array(neg_cont, dtype=FLOAT_DTYPE),
                     "clone": clone,
                     # If clone is not contiuous, then there will be problems with indexing
                     "clone_continuous": None if clone is None else jnp.searchsorted(jnp.unique(clone), clone),
                     "sigma": None if sigma is None else jnp.array(sigma, dtype=FLOAT_DTYPE),
                     }

        self.mode = mode
        self.alpha_model = alpha_model

    def _init_kmeans(self, scale_factor=1.0, outlier_threshold=4.0) -> Dict:
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
        x_log = np.log(x + 1)  # Transform to log scale, roughly normal distributed
        zscore = (x_log - x_log.mean()) / x_log.std()
        x_no_outliers = x[zscore < outlier_threshold]
        x_no_outliers = x_no_outliers.sort()
        diffs = np.diff(x_no_outliers, prepend=x_no_outliers[0])
        # TODO What value to define a huge gap?
        huge_gap_indices = np.where(diffs > x.max() / 3)[0]
        if len(huge_gap_indices) > 0:
            first_gap_index = huge_gap_indices[0]
            x_no_outliers = x_no_outliers[:first_gap_index]

        clone = self.data.get("clone_continuous", None)
        sigma = self.data.get("sigma", None)
        n_clusters = 2  # KMeans with 2 clusters

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, init=np.vstack([np.min(x_no_outliers), np.max(x_no_outliers)]), n_init="auto").fit(x_no_outliers.reshape(-1, 1))
        labels = kmeans.predict(x.reshape(-1, 1))

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

        # Set mode-specific parameters (mode "C" for clonotypes)
        if clone is not None:
            unique_clones = jnp.unique(clone)
            clone_proportions = []
            clone_means = []
            clone_means_per_cluster = []
            clone_variances = []
            clone_variances_per_cluster = []

            for unique_clone in unique_clones:
                clone_points = x[clone == unique_clone]
                clone_labels = labels[clone == unique_clone]

                # Clone-specific proportions
                clone_cluster_counts = np.bincount(clone_labels, minlength=2)[sorted_indices]
                clone_proportions.append(clone_cluster_counts / len(clone_points))

                # Clone-specific means and variances
                clone_means.append(np.mean(clone_points))
                clone_means_per_cluster.append([np.mean(clone_points[clone_labels == 0]), np.mean(clone_points[clone_labels == 1])])
                clone_variances.append(np.var(clone_points))
                clone_variances_per_cluster.append([np.var(clone_points[clone_labels == 0]), np.var(clone_points[clone_labels == 1])])

            cluster_proportions = np.array(clone_proportions)
            clone_means = np.array(clone_means)
            clone_variances = np.array(clone_variances)
            clone_means_per_cluster = np.array(clone_means_per_cluster)
            clone_variances_per_cluster = np.array(clone_variances_per_cluster)

        # Update model configuration with calculated priors
        kmeans_dict.update({
            "z": labels,
            "cluster_means": cluster_means,  # Mean for each cluster
            "cluster_variances": cluster_variances,  # variance for each cluster
            "cluster_proportion": cluster_proportions,
            "tau_concentration_prior": cluster_proportions * 10 + 1,  # Concentration for Dirichlet prior
            "clone_means": clone_means if clone is not None else None,  # Mean for each clonotype
            "clone_variances": clone_variances if clone is not None else None,  # Variance for each clonotype
            "clone_means_per_cluster": clone_means_per_cluster if clone is not None else None,  # Mean for each clonotype per cluster
            "clone_variances_per_cluster": clone_variances_per_cluster if clone is not None else None,  # Variance for each clonotype per cluster
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
        clone = self.data["clone_continuous"]
        sigma = self.data["sigma"]
        s =  jnp.ones(x.shape[0]) if self.data["s"] is None else self.data["s"]

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
                                          npd.NegativeBinomial2(mean=s*q[0], concentration=alpha),
                                          obs=x_neg)
                elif self.mode == "C":
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=s*q[0],
                                                                concentration=alpha[clone]),
                                          obs=x_neg)
                else:
                    yhat_neg = npy.sample("yhat_neg",
                                          npd.NegativeBinomial2(mean=s*q[0], concentration=alpha[0]),
                                          obs=x_neg)

            # target pMHC
            if self.mode == "C":
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=s[:,None]*q,
                                                                         concentration=alpha[clone].reshape(
                                                                             (N_sample, 1))
                                                                         )
                                                )
            else:
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=s[:,None]*q,
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
        self._kmeans_dict = self._init_kmeans(scale_factor=scale_factor,
                                              outlier_threshold=kwargs['outlier_threshold']
                                              if 'outlier_threshold' in kwargs else 4.0)
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
        s = jnp.ones(x.shape[0]) if self.data["s"] is None else self.data["s"]
        x_neg = self.data["x_neg"]
        clone = self.data["clone_continuous"]
        sigma = self.data["sigma"]
        N_sample = x.shape[0]
        c_nof = np.unique(clone).size if clone is not None else 0
        K = 2

        # Extract hyperpriors
        mu_w_mean_prior = model_config["mu_w_mean_prior"]
        mu_w_var_prior = model_config["mu_w_var_prior"]
        cluster_means = model_config["cluster_means"]
        cluster_variances = model_config["cluster_variances"]
        var_hp = model_config["var_hyperprior"]  # Variance for priors if using alpha_model = kmeans
        tau_concentration_prior = model_config["tau_concentration_prior"]
        overdispersion_scale_prior = model_config["overdispersion_scale_prior"]

        clone_means = model_config.get("clone_means", None)  # Mean for each clonotype
        clone_variances = model_config.get("clone_variances", None)
        cluster_proportion = model_config.get("cluster_proportion", None)  # Variance for priors if using alpha_model = kmeans

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
            mean_deltas = jnp.array([max(cluster_means[0], 1e-1), cluster_means[1] - cluster_means[0]])
            var_deltas = jnp.array([max(cluster_variances[0], 1e-1), max(cluster_variances[1] - cluster_variances[0], 1)])

            # Convert kmeans parameters to lognormal parameters with target mean and variance
            # NB mean parameter: q_prior ~ LogNormal(mu_q, sigma_q), with cluster means and variances
            sigma2_q_prior = jnp.log(var_deltas / mean_deltas ** 2 + 1)
            sigma_q_prior = jnp.sqrt(sigma2_q_prior)
            mu_q_prior = jnp.log(mean_deltas) - sigma2_q_prior / 2

            # Sample delta_q from lognormal distribution and cumsum to create ordered q
            with npy.plate("cluster_axis", K):
                delta_q = npy.sample("q", npd.LogNormal(loc=mu_q_prior, scale=sigma_q_prior))
            q = jnp.cumsum(delta_q, axis=0)

            # NB concentration parameter: alpha = q^2 / (q * overdispersion - q), overdispersion ~ HalfCauchy(1) + 1
            overdispersion_prior_dist = npd.HalfCauchy(overdispersion_scale_prior)

            if self.mode == "C":
                # For each clonotype, we have one alpha parameter, weight should adjust itself so that one alpha
                # parameter is actually used
                with npy.plate("clone_axis", c_nof):
                    overdispersion = npy.sample("overdispersion", overdispersion_prior_dist) + 1
                    q_weighted = (w * q).mean(1)
                alpha = npy.deterministic("alpha", q_weighted ** 2 / (q_weighted * overdispersion - q_weighted))
            else:
                # For each mixture component, we have one alpha parameter
                with npy.plate("cluster_axis", K):
                    overdispersion = npy.sample("overdispersion", overdispersion_prior_dist) + 1
                alpha = npy.deterministic("alpha", q**2 / (q * overdispersion - q))

        elif self.alpha_model == "kmeans":
            # NB mean parameter: q ~ LogNormal(mu_q, sigma_q),
            # with prior on cluster means and hyperprior variances
            mean_deltas = jnp.array([max(cluster_means[0], 1e-1), cluster_means[1] - cluster_means[0]])
            # Cumsum also cumsums the variances, use a small value to be added
            var_hp_deltas = jnp.array([var_hp, 1])

            sigma2_q_hp = jnp.log(var_hp_deltas / mean_deltas ** 2 + 1)
            sigma_q_hp = jnp.sqrt(sigma2_q_hp)
            mu_q_prior = jnp.log(mean_deltas) - sigma2_q_hp / 2

            with npy.plate("cluster_axis", K):
                delta_q = npy.sample("q", npd.LogNormal(loc=mu_q_prior, scale=sigma_q_hp))
            q = jnp.cumsum(delta_q, axis=0)

            # NB concentration parameter alpha: convert kmeans variance priors to Gamma parameters,
            # alpha ~ Gamma(a, b),
            # with a, b so that mean = cluster_variance and var = hyperprior_variances of the LogNormal

            # Convert kmeans cluster variance to alpha parameter (also dependent on cluster mean)
            if self.mode == "C":
                # Use mean and variance of each clone as prior, alpha_prior.shape = (c_nof)
                alpha_prior = clone_means**2 / (np.maximum(clone_variances, 1e-1) - clone_means - 1e-8)
            elif self.mode == "I":
                # Use kmeans cluster mean and variance as prior, alpha_prior.shape = (2)
                alpha_prior = cluster_means**2 / (np.maximum(cluster_variances, 1e-1) - cluster_means)

            # In case of underdispersion (negative alpha), set to a high number, so var ~ mean
            alpha_prior[alpha_prior <= 0] = 100

            # Set prior to 1 if between 0 and 1 to avoid numerical issues
            alpha_prior[(alpha_prior > 0) & (alpha_prior < 1)] = 1

            # LogNormal
            # sigma2_alpha_hp = jnp.log(var_hp / alpha_prior ** 2 + 1)
            # sigma_alpha_hp = jnp.sqrt(sigma2_alpha_hp)
            #
            # mu_alpha_prior = jnp.log(alpha_prior) - sigma2_alpha_hp / 2

            # if self.mode == "C":
            #     with npy.plate("clone_axis", c_nof):
            #         alpha = npy.sample("alpha", npd.LogNormal(loc=mu_alpha_prior, scale=sigma_alpha_hp))
            # elif self.mode == 'I':
            #     with npy.plate("cluster_axis", K):
            #         alpha = npy.sample("alpha", npd.LogNormal(loc=mu_alpha_prior, scale=sigma_alpha_hp))

            # Gamma distribution
            # compute Gamma parameters
            a = alpha_prior ** 2 / var_hp
            b = alpha_prior / var_hp

            if self.mode == "C":
                with npy.plate("clone_axis", c_nof):
                    # shape = (c_nof, 1)
                    alpha = npy.sample("alpha", npd.Gamma(concentration=a, rate=b))
            elif self.mode == 'I':
                with npy.plate("cluster_axis", K):
                    # shape = (2, )
                    alpha = npy.sample("alpha", npd.Gamma(concentration=a, rate=b))

        else:
            raise NotImplementedError(f" {self.alpha_model} not implemented")

        # Sample from the mixture model
        with npy.plate("sample_axis", N_sample):
            if x_neg is not None:
                if self.mode == "C":
                    yhat_neg = npy.sample("yhat_neg", obs=x_neg,
                                          fn=npd.NegativeBinomial2(mean=s*q[0],
                                                                concentration=alpha[clone]),)

                else:
                    yhat_neg = npy.sample("yhat_neg", obs=x_neg,
                                          fn=npd.NegativeBinomial2(mean=s*q[0], concentration=alpha[0]),)

            # target pMHC
            if self.mode == "C":
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=s[:,None]*q, concentration=alpha[clone, None]))
            else:
                mixture = npd.MixtureSameFamily(z, npd.NegativeBinomial2(mean=s[:,None]*q, concentration=alpha))

            yhat = npy.sample("yhat", mixture, obs=x)

            # Membership probability of each sample
            log_probs = mixture.component_log_probs(yhat)
            p = npy.deterministic("log_p", log_probs - logsumexp(log_probs, axis=-1, keepdims=True))
