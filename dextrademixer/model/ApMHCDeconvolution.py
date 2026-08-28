"""
The interface shared by all pMHC deconvolution methods.

`ApMHCDeconvolution` fixes the call order every method follows and holds the parts that do not
depend on the model: resolving the supported input containers into counts and annotation
(`as_counts`), validating them (`_check_parameters`), and turning posterior probabilities into class
assignments by threshold or local-FDR control (`_predict_posterior_class`). DextraDemixer and the
comparison baselines BEAM, ICON and ITRAP all build on it.
"""
from __future__ import annotations

import abc

from typing import TYPE_CHECKING, Tuple, Union

import anndata as ad
import mudata as md
import pandas as pd

from jax import lax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax._src.typing import Array

Data = Union[md.MuData, ad.AnnData, pd.DataFrame]


class ApMHCDeconvolution:
    """
    Common interface of the pMHC deconvolution methods, i.e. DextraDemixer and the comparison
    baselines. It fixes the call order `preprocess_model_data` -> `fit` -> `predict_posterior_class`
    and provides the model-agnostic parts of it.
    """

    @abc.abstractmethod
    def preprocess_model_data(self,
                              data: Data,
                              pmhc_key: str,
                              gex_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_key: str = None,
                              **kwargs):
        """
        Extracts the counts and the annotation a method needs from `data`, see `as_counts`.

        Args:
            data: the pMHC counts, as a MuData, an AnnData or a cells x features DataFrame.
            pmhc_key: the pMHC count column to deconvolve.
            gex_key: the MuData modality holding the counts.
            neg_ctrl_key: (Optional) the negative control count column.
            ir_key: the MuData AIRR module key.
            ir_clone_key: (Optional) the `obs` column holding clonotype ids.
            kwargs: method-specific extras.
        """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fits the method on the data prepared by `preprocess_model_data`.

        Returns:
            self, so that `predict_posterior_class` can be chained onto the call.
        """

    @abc.abstractmethod
    def predict_posterior_class(self,
                                threshold: float = None,
                                target_fdr: float = None
                                ) -> Tuple[Array, Array]:
        """
        Assigns each cell to the binding or non-binding class.

        Args:
            threshold: (Optional) probability in [0,1] above which a cell is called a binder.
            target_fdr: (Optional) FDR to control instead of using a fixed threshold. Mutually
                        exclusive with `threshold`.

        Returns:
            A tuple (p, assignment) of per-cell binding probabilities and 0/1 assignments.
        """

    @staticmethod
    def _predict_posterior_class(p: Array,
                                 threshold: float = None,
                                 target_fdr: float = None
                                 ) -> Array:
        """
        Turns per-cell binding probabilities into 0/1 assignments, either at a fixed threshold or
        at the largest threshold whose estimated FDR stays below `target_fdr`.

        Args:
            p: per-cell posterior probability of binding, shape (n_cells,).
            threshold: (Optional) probability in [0,1] above which a cell is called a binder.
            target_fdr: (Optional) FDR to control instead. Mutually exclusive with `threshold`;
                        if neither is given, a threshold of 0.5 is used.

        Returns:
            The 0/1 class assignment, shape (n_cells,).

        Raises:
            ValueError: if both `threshold` and `target_fdr` are given, or either is outside [0,1].
        """
        if threshold is not None and target_fdr is not None:
            raise ValueError("Please specify either a manual `threshold` or a `target_fdr` but not both.")

        if threshold is not None and not (0 <= threshold <= 1):
            raise ValueError(f"`threshold`must be in [0,1] but was {threshold}")

        if target_fdr is not None and not (0 <= target_fdr <= 1):
            raise ValueError(f"`target_fdr`must be in [0,1] but was {target_fdr}")

        if threshold is None and target_fdr is None:
            threshold = 0.5

        # posterior probability of belonging to the binding class
        if target_fdr is not None:
            N = p.shape[0]

            # Calculate the local FDR (1 - p)
            lfdr = 1 - p

            sorted_indices = jnp.argsort(p)[::-1]
            sorted_p = p[sorted_indices]
            sorted_lfdr = lfdr[sorted_indices]

            cumulative_lfdr = jnp.cumsum(sorted_lfdr)
            cumulative_count = jnp.arange(1, N + 1)

            # Estimated FDR for each possible threshold
            estimated_fdr = cumulative_lfdr / cumulative_count

            # Find the largest index k such that estimated_fdr[k] <= target_fdr
            valid_thresholds = estimated_fdr <= target_fdr
            max_k = jnp.max(jnp.where(valid_thresholds, jnp.arange(N), -1))
            threshold = lax.cond(max_k >= 0, lambda: sorted_p[max_k], lambda: 1.0)

        assignment = (p >= threshold).astype("int32")
        return assignment


    @staticmethod
    def as_counts(data: Data, gex_key: str = "gex", ir_key: str = "airr") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalizes any supported input into a (counts, obs) pair of DataFrames, so that count
        columns are addressed by `counts[key]` and per-cell annotation by `obs[key]`:

        - MuData: counts from the `gex_key` modality, obs from the `ir_key` modality
        - AnnData: counts from `X`, obs from `obs`
        - DataFrame (cells x features): obs is the frame itself, so clonotypes can be a column

        Args:
            data: a MuData, an AnnData or a cells x features DataFrame.
            gex_key: the MuData modality holding the counts, unused for the other types.
            ir_key: the MuData modality holding the annotation, unused for the other types.

        Returns:
            A tuple (counts, obs) of DataFrames sharing the cell order of `data`.

        Raises:
            TypeError: if `data` is of an unsupported type.
        """
        if isinstance(data, md.MuData):
            counts, _ = ApMHCDeconvolution.as_counts(data.mod[gex_key])
            return counts, data.mod[ir_key].obs if ir_key in data.mod else data.obs
        if isinstance(data, ad.AnnData):
            return data.to_df(), data.obs
        if isinstance(data, pd.DataFrame):
            return data, data
        raise TypeError(f"unsupported input type {type(data).__name__}, expected a MuData, an "
                        f"AnnData or a cells x features DataFrame")

    @staticmethod
    def _check_parameters(x, x_neg, clone_id):
        """
        Checks consistency of the input data before initializing the model.

        Args:
            x: pMHC UMI counts, shape (n_cells,).
            x_neg: (Optional) negative control counts, expected shape (n_cells,).
            clone_id: (Optional) clonotype ids, expected shape (n_cells,).

        Raises:
            ValueError: if `x` contains NaNs or if `x_neg`/`clone_id` do not match its length.
        """
        N = x.shape[0]

        if jnp.isnan(x).any():
            raise ValueError("Input data `x` contains NaN values. Please remove them before fitting the model.")

        if clone_id is not None:
            if clone_id.shape[0] != N:
                raise ValueError(f"`clone_id` and count data `x` require the same size but got "
                                 f"{clone_id.shape[0]} and {N}")

        if x_neg is not None:
            N_neg = x_neg.shape[0]

            if N_neg != N:
                raise ValueError(f"x_neg must have the same size than x but got {N_neg} vs {N}.")

