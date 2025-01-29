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


def threshold_assign_pmhc(mdata: md.MuData,
                          threshold: float,
                          threshold_type: str = "absolute",
                          pmhc_keys: Union[str, List[str]] = None,
                          total_normalization: bool = False,
                          target_sum: float = None,
                          z_score_normalization: bool = False,
                          gex_key: str = "gex",
                          neg_ctrl_key: str = None,
                          ir_key: str = "airr",
                          ir_clone_key: str = None,
                          inplace=False):
    """
    Assigns dextramer-specificities based on specified threshold.
    Depending on additional information provided different assignment strategies are applied.


    Args:
        mdata: A Mudata containing only dextramer counts and clonotype information
        threshold: A UMI count, or relative threshold to determine dextramer-specificity
        threshold_type: A string specifiying whether the threshold is absolut or relative. if relative than X in gex_key
                        will be normalized by the column means
        pmhc_keys (Optional): A string or list of strings indicating the pMHC columns in `gex_key` modality`s `X` which
                              should be deconvolved. If None is given, the full X is used
        total_normalization: boolean whether or not normalization of each cell by total counts over all pMHCs
                             (including negative control) should be applied, so that every cell has the same total count
                              after normalization.
        target_sum: If None, after normalization, each observation (cell) has a total count equal to the median of total
                    counts for observations (cells) before normalization.
        z_score_normalization: z-score normalize within pMHC across cells
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
    N = gex.shape[0]

    if pmhc_keys is None:
        pmhc_keys = gex.var.index

    if neg_ctrl_key is not None and neg_ctrl_key not in pmhc_keys:
        pmhc_keys.append(neg_ctrl_key)

    gex_filtered = gex[:, pmhc_keys]

    if total_normalization:
        x = sc.pp.normalize_total(gex_filtered, target_sum=target_sum, inplace=False)['X']
    else:
        x = gex_filtered.X.toarray()

    if z_score_normalization:
        x = (x - jnp.nanmean(x, axis=0, keepdims=True)) / jnp.nanstd(x, axis=0, keepdims=True)

    neg_ctrl_key_idx = gex.var.index.tolist().index(neg_ctrl_key) if neg_ctrl_key else None
    x_neg = x[:, neg_ctrl_key_idx].reshape((N,)) if neg_ctrl_key else None
    x = jnp.delete(x, neg_ctrl_key_idx, axis=1) if neg_ctrl_key else x
    c = air.obs[ir_clone_key].to_numpy().astype("int32") if ir_clone_key is not None else None

    if threshold_type == "relative" and neg_ctrl_key is None:
        x = x / (x.sum(axis=1, keepdims=True)+1e-8)
    elif threshold_type == "relative" and neg_ctrl_key is not None:
        x = x / (jnp.nanmax(x_neg)+1e-8)
    elif threshold_type == "absolut" and neg_ctrl_key is not None:
        x = x - jnp.nanmax(x_neg)

    assignment = ((x == jnp.max(x, axis=1, keepdims=True)) & (x >= threshold)).astype("uint8")

    if inplace:
        mdata.mod[gex_key].obsm["pMHC_assignment"] = assignment
    else:
        return assignment


def icon_assign_pmhc(mdata: md.MuData,
                      neg_ctrl_key: str,
                      ir_clone_key: str,
                      threshold: float = 0,
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
    N = gex.shape[0]

    if pmhc_keys is None:
        pmhc_keys = gex.var.index

    neg_ctrl_key_idx = gex.var.index(neg_ctrl_key)

    X = gex[:, pmhc_keys].X.toarray()
    x_neg = gex.X[:, neg_ctrl_key_idx].max() if neg_ctrl_key else None
    c = air.obs[ir_clone_key].to_numpy().astype("int32") if ir_clone_key is not None else None

    # Subtract background noise
    E = X - x_neg
    E = E.at[E < 0].set(0)

    # calc pMHC ratio per cell
    C = E / (E.sum(axis=1, keepdims=True)+1)

    # raw assignment with UMI > 0
    rA = (E > 0).astype("int32")

    # calc clonotype purity
    R = calculate_pmhc_clonal_purity(rA, c)

    S = jnp.log(E + 0.01) * C ** 2 * R
    S = jnp.nan_to_num(S)
    S = S.at[S < 0].set(0)

    # pMHC-wise log-ratio normalization per cell
    colSum = S.sum(axis=1, keepdims=True)
    colSum = colSum.at[colSum <= 0].set(1)
    S = S / colSum

    # cell-wise z-score normalization
    S = (S - jnp.nanmean(S)) / jnp.nanstd(S)
    S = jnp.nan_to_num(S, nan=jnp.nanmin(S))

    assignment = (S > threshold).astype("uint8")

    if inplace:
        mdata.mod[gex_key].obsm["pMHC_assignment"] = assignment
    else:
        return assignment
