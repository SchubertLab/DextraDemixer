import warnings
from typing import List, Union

import pandas as pd
import mudata as md
import scanpy as sc
import scipy.stats
from scipy.stats import zscore

import jax
import jax.lax
import jax.numpy as jnp

from dextrademixer.utils import calculate_pmhc_clonal_purity


def icon_assign_pmhc(mdata: md.MuData,
                     ir_clone_key: str,
                     neg_ctrl_key: str = None,
                     threshold: float = 0,
                     bg_noise: float = None,
                     pmhc_keys: Union[str, List[str]] = None,
                     gex_key: str = "gex",
                     ir_key: str = "airr",
                     inplace=False):
    """
    implements the ICON assignment procedure

    Args:
        mdata: A Mudata containing only dextramer counts and clonotype information
        threshold: A UMI count, or relative threshold to determine dextramer-specificity
        threshold_type: A string specifiying whether the threshold is absolut or relative. if relative than X in gex_key
                        will be normalized by the column means
        pmhc_keys (Optional): A string or list of strings indicating the pMHC columns in `gex_key` modality`s `X` which should be
                   deconvolved. If None is given, the full X is used
        gex_key: the MuData transcriptome module key
        neg_ctrl_key: (Optional) a string specifying the negative control column in `gex_key` modality`s `X`
        ir_key: the MuData AIRR module key
        ir_clone_key: (Optional) a string specifying the field in `obs` of `ir_key` that holds clonotype ids
        inplace: boolean indicating whether assignment should be stored in mdata on `gex_key` `obsm`
        kwargs: dictionary of additional information pasted to the Model object (used for custom model prior)


    Returns: An array of pMHC assignments per cell, or modifies the mdata object adding an obsm matrix at `gex_key`
    """
    gex = mdata.mod[gex_key]
    air = mdata.mod[ir_key]

    if pmhc_keys is None:
        pmhc_keys = gex.var_names

    if bg_noise is None and neg_ctrl_key is None:
        bg_noise = 10

    X = jnp.array(gex[:, pmhc_keys].X.toarray())
    x_neg = gex.X[:, neg_ctrl_key].max() if bg_noise is None else bg_noise
    c = air.obs[ir_clone_key].to_numpy().astype("int32")

    # Subtract background noise
    E = X - x_neg
    E = E.at[E < 0].set(0)

    # calc pMHC ratio per cell
    C = E / (E.sum(axis=1, keepdims=True) + 1)

    # raw assignment with UMI > 0
    rA = (E > 0).astype("int32")

    # calc clonotype purity
    R = calculate_pmhc_clonal_purity(rA, c)

    S = jnp.log(E + 0.01) * (C ** 2) * R
    S = jnp.nan_to_num(S)
    S = S.at[S < 1].set(0)

    # pMHC-wise log-ratio normalization per cell
    colSum = S.sum(axis=1, keepdims=True)
    colSum = colSum.at[colSum <= 0].set(1)
    S = S / colSum

    # cell-wise z-score normalization
    S = (S - jnp.nanmean(S, axis=0, keepdims=True)) / jnp.nanstd(S, axis=0, keepdims=True)
    S = jnp.nan_to_num(S, nan=jnp.nanmin(S))

    assignment = (S > threshold).astype("uint8")

    if inplace:
        mdata.mod[gex_key].obsm["pMHC_assignment"] = assignment
    else:
        return assignment
