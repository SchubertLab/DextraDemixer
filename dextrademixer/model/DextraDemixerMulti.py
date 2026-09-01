"""
Multi-pMHC deconvolution: one independent `DextraDemixer` per pMHC, plus cross-pMHC resolving resulting cells x pMHC probability matrix into unique assignments via `max_prob`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd

from dextrademixer.model.ApMHCDeconvolution import ApMHCDeconvolution, Data
from dextrademixer.model.DextraDemixer import DextraDemixer

if TYPE_CHECKING:
    from jax._src.typing import Array


class DextraDemixerMulti(ApMHCDeconvolution):
    r"""
    Runs `DextraDemixer` over several pMHCs and summarizes the results across them.

    Each pMHC gets its own `DextraDemixer`, fit independently; the models are reachable as
    `self.demixers[pmhc_key]` if a single fit needs inspecting or plotting. Call order is the same
    as for `DextraDemixer`: `preprocess_model_data` -> `fit_svi` -> `predict_posterior_class`, or
    `fit` for the first two in one go.

    On top of the per-pMHC results, `max_prob=True` turns the cells x pMHC assignment matrix into
    unique assignments: among the pMHCs whose probability clears the threshold, keep the highest. It acts
    on whatever probability was thresholded, so `clonotype_median_p` picks the level.
    """

    def __init__(self, **demixer_kwargs):
        """
        Args:
            demixer_kwargs: passed to every per-pMHC `DextraDemixer`, e.g. `model_type`,
                            `overdispersion_scale_prior` or `alpha_offset`.
        """
        super().__init__()
        self._demixer_kwargs = demixer_kwargs
        self.demixers: Dict[str, DextraDemixer] = {}
        self.pmhc_keys: List[str] = []
        self.obs_names = None
        self.counts = None

    def preprocess_model_data(self,
                              data: Data,
                              pmhc_keys: List[str],
                              pmhc_modality_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_modality_key: str = "airr",
                              ir_clone_key: str = None,
                              size_factor_keys: Union[bool, List[str]] = None,
                              outlier_z_score: float = 100):
        """
        Preprocesses the data and initializes one model per pMHC.

        Args:
            data: the dextramer counts, as a MuData, an AnnData or a cells x features DataFrame.
                  See `as_counts` for where counts and annotation are read from in each case;
                  `pmhc_modality_key`/`ir_modality_key` are only used for MuData.
            pmhc_keys: the pMHC count columns to deconvolve. `neg_ctrl_key` is dropped from the
                       list if present.
            pmhc_modality_key: the MuData modality holding the counts
            neg_ctrl_key: (Optional) the negative control count column
            ir_modality_key: the MuData AIRR module key
            ir_clone_key: (Optional) the `obs` column that holds clonotype ids (ints or strings)
            size_factor_keys: (Optional) which pMHC columns to compute size factors from, True for
                             all of them. Shared across the pMHCs, as every delegate derives them
                             from the same columns
            outlier_z_score: cells more than this many standard deviations from the mean count are
                        held out of the fit but still scored by `predict_posterior_class`. None
                        disables the filtering

        Raises:
            ValueError: if `pmhc_keys` is empty after removing the negative control.
        """
        self.pmhc_keys = [k for k in pmhc_keys if k != neg_ctrl_key]
        if not self.pmhc_keys:
            raise ValueError("`pmhc_keys` is empty after removing `neg_ctrl_key`.")

        counts, _ = self.as_counts(data, pmhc_modality_key, ir_modality_key)
        self.obs_names = counts.index
        self.counts = counts[self.pmhc_keys]  # only needed to break ties in `_resolve`

        self.demixers = {}
        for pmhc_key in self.pmhc_keys:
            demixer = DextraDemixer(**self._demixer_kwargs)
            demixer.preprocess_model_data(data, pmhc_key=pmhc_key, pmhc_modality_key=pmhc_modality_key,
                                          neg_ctrl_key=neg_ctrl_key, ir_modality_key=ir_modality_key,
                                          ir_clone_key=ir_clone_key,
                                          size_factor_keys=size_factor_keys,
                                          outlier_z_score=outlier_z_score)
            self.demixers[pmhc_key] = demixer

    def fit_svi(self, progress_bar: bool = True, **svi_kwargs) -> None:
        """
        Fits every pMHC with SVI, see `DextraDemixer.fit_svi` for the inference settings.

        Args:
            progress_bar: whether to show a progress bar per pMHC, labelled with the pMHC key
            svi_kwargs: passed to `DextraDemixer.fit_svi`, e.g. `maxiter`, `n_inits`, `rng_key`

        Raises:
            RuntimeError: if `preprocess_model_data` has not been called yet.
        """
        if not self.demixers:
            raise RuntimeError("Model is not initialized. Please call `preprocess_model_data` first.")

        for i, (pmhc_key, demixer) in enumerate(self.demixers.items(), start=1):
            if progress_bar:
                print(f"Fitting {i}/{len(self.demixers)}: {pmhc_key}")
            demixer.fit_svi(progress_bar=progress_bar, **svi_kwargs)

    def fit(self, data: Data, *, pmhc_keys: List[str], pmhc_modality_key: str = "gex",
            neg_ctrl_key: str = None, ir_modality_key: str = "airr", ir_clone_key: str = None,
            size_factor_keys: Union[bool, List[str]] = None, outlier_z_score: float = 100,
            **svi_kwargs) -> "DextraDemixerMulti":
        """
        `preprocess_model_data` followed by `fit_svi`. The recommended entry point.

        Args:
            data: the dextramer counts, as a MuData, an AnnData or a cells x features DataFrame
            pmhc_keys: the pMHC count columns to deconvolve
            pmhc_modality_key: the MuData modality holding the counts
            neg_ctrl_key: (Optional) the negative control count column
            ir_modality_key: the MuData AIRR module key
            ir_clone_key: (Optional) the `obs` column that holds clonotype ids
            size_factor_keys: (Optional) which pMHC columns to compute size factors from
            outlier_z_score: see `preprocess_model_data`
            svi_kwargs: passed to `DextraDemixer.fit_svi`

        Returns:
            self, so that `predict_posterior_class` can be chained onto the call
        """
        self.preprocess_model_data(data, pmhc_keys=pmhc_keys, pmhc_modality_key=pmhc_modality_key,
                                   neg_ctrl_key=neg_ctrl_key, ir_modality_key=ir_modality_key,
                                   ir_clone_key=ir_clone_key, size_factor_keys=size_factor_keys,
                                   outlier_z_score=outlier_z_score)
        self.fit_svi(**svi_kwargs)
        return self

    def predict_posterior_class(self,
                                threshold: float = None,
                                target_fdr: float = None,
                                cred_intvl: float = None,
                                clonotype_median_p: bool = False,
                                clone_id: Array = None,
                                max_prob: bool = False,
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the per-pMHC binder assignments, optionally resolved into unique calls per cell.

        The first five arguments are handed to `DextraDemixer.predict_posterior_class` unchanged and
        apply to every pMHC alike; `max_prob` then acts across pMHCs.

        Args:
            threshold: (Optional) a threshold in [0,1] determining binder based on inferred
                       posterior class probabilities
            target_fdr: (Optional) the FDR to control instead of a fixed threshold. Controlled per
                        pMHC, not across them
            cred_intvl: (Optional) instead of the summarized class probability, estimate a
                        distribution over Pr(FDR(t)<=alpha|posterior)>=cred_intvl
            clonotype_median_p: (Optional) replace each cell's probability by the median within its
                        clonotype before assigning
            clone_id: (Optional) clonotype id per cell, overriding the one given at preprocessing
            max_prob: whether to narrow multiple calls per cell down to the pMHC with the highest
                      probability among those clearing the threshold. Acts on the same probability
                      that was thresholded, so `clonotype_median_p` decides whether that is the
                      per-cell or the per-clonotype one. Equal probabilities go to the higher
                      multimer count, then to the first pMHC

        Returns:
            A tuple (p_pred, assignment) of cells x pMHC DataFrames, with p_pred the posterior probability of
            binding and assignment the 0/1 class assignment decision.

        Raises:
            RuntimeError: if the models have not been fit yet.
            ValueError: if `clonotype_median_p` is set but no clonotypes are available.
        """
        if not self.demixers:
            raise RuntimeError("Model has not been fit yet. Please call first `fit` or `fit_svi`.")

        results = {k: d.predict_posterior_class(threshold=threshold, target_fdr=target_fdr,
                                                cred_intvl=cred_intvl,
                                                clonotype_median_p=clonotype_median_p,
                                                clone_id=clone_id)
                   for k, d in self.demixers.items()}
        p_pred = pd.DataFrame({k: np.asarray(v[0]) for k, v in results.items()}, index=self.obs_names)
        assignment = pd.DataFrame({k: np.asarray(v[1]) for k, v in results.items()},
                                  index=self.obs_names)

        return p_pred, self._resolve(p_pred, assignment, self.counts, max_prob=max_prob)

    def _resolve(p_pred: pd.DataFrame,
                 assignment: pd.DataFrame,
                 counts: pd.DataFrame,
                 max_prob: bool = False) -> pd.DataFrame:
        """
        Keeps, per cell, only the called pMHC with the highest probability.

        Equal probabilities go to the higher multimer count, then to the first pMHC. A cell without
        a call stays unassigned.

        Args:
            p_pred: cells x pMHC posterior probabilities.
            assignment: cells x pMHC 0/1 assignments, same shape and columns as `p_pred`.
            counts: cells x pMHC multimer counts, same shape and columns as `p_pred`.
            max_prob: whether to resolve at all. False returns `assignment` unchanged.

        Returns:
            The narrowed 0/1 assignment matrix.
        """
        if not max_prob:
            return assignment

        p = p_pred.where(assignment.astype(bool))  # only called pMHCs compete
        best = p.eq(p.max(axis=1), axis=0).to_numpy()
        winner = np.where(best, counts.to_numpy(), -np.inf).argmax(axis=1)  # first wins on equal counts

        out = np.zeros(assignment.shape, dtype=int)
        out[np.arange(len(out)), winner] = best.any(axis=1)  # nothing called -> nothing assigned
        return pd.DataFrame(out, index=assignment.index, columns=assignment.columns)

    def summary(self) -> pd.DataFrame:
        """
        Concatenates the per-pMHC `DextraDemixer.summary()` tables, keyed by pMHC.

        Returns:
            The arviz summary of every fit, with the pMHC key as the outer index level.
        """
        return pd.concat([d.summary() for d in self.demixers.values()],
                         keys=list(self.demixers), names=["pMHC"])
