from typing import List, Union
import numpy as np
import pandas as pd
import mudata as md
import anndata as ad


def icon_assign_pmhc(adata: Union[md.MuData, ad.AnnData],
                     ir_clone_key: str,
                     neg_ctrl_key: str = None,
                     threshold: float = 0,
                     bg_noise: float = None,
                     bg_noise_quantile: float = 0.975,
                     pmhc_keys: Union[str, List[str]] = None,
                     dex_key: str = "dex",
                     inplace=False,
                     faithful: bool = False,
                     ):
    """
    implements the ICON assignment procedure
    requires clonal information and dextramer counts, and optionally a negative control column to estimate background noise.

    Args:
        adata: A MuData object containing only dextramer counts and clonotype information,
            or an AnnData object containing the dextramer counts and clonotype information in the specified obsm and obs keys.
        threshold: A relative threshold to determine dextramer-specificity
        bg_noise: (Optional) A value to substract from dextramer counts to account for background noise. 
            If None is given, the bg_noise_quantile of the negative control column is used if specified, otherwise 10.
        pmhc_keys (Optional): A string or list of strings indicating the pMHC columns in `dex_key` modality which should be
            deconvolved. If None is given, the full matrix is used, excluding the negative control if specified.
        dex_key: the dextramer signal MuData module key, or the obsm key if adata is an AnnData object
        neg_ctrl_key: (Optional) a string specifying the negative control column in the `dex_key` matrix.
        ir_clone_key: A string specifying the field in `obs` that holds clonotype ids. 
            If in the immune receptor modality of a mudata object, should be `ir_key:clone_key`.
        inplace: boolean indicating whether assignment should be stored in `obsm`
        faithful: boolean indicating whether to use the original ICON procedure (True) or a debuged version based on the paper description

    Returns: An array of pMHC assignments per cell, or modifies the adata object adding an obsm matrix at `dex_key`
    """
    # check if clone key contains NA values
    if adata.obs[ir_clone_key].isna().sum() > 0:
        raise ValueError(f"NA values found in clone key {ir_clone_key} of adata.obs. ICON works only for cells with TCR information. Please filter the object.")
    c = adata.obs[ir_clone_key].to_numpy().astype("int32")

    # get dextramer counts
    if isinstance(adata, md.MuData):
        is_mudata = True
        dex = adata.mod[dex_key]
        dex = pd.DataFrame(dex.X.toarray(), index=dex.obs_names, columns=dex.var_names)
    elif isinstance(adata, ad.AnnData):
        is_mudata = False
        dex = adata.obsm[dex_key]

    if pmhc_keys is None:
        X = dex.loc[:,dex.columns != neg_ctrl_key].values

    # get background noise
    if bg_noise is None:
        bg_noise = np.quantile(dex.loc[:, neg_ctrl_key], q=bg_noise_quantile) if neg_ctrl_key is not None else 10

    # substract background
    E = np.maximum(0, X - bg_noise)

    # calc pMHC ratio per cell
    if faithful:
        # +1 in the denominator can have large effects
        C = E / (E.sum(axis=1, keepdims=True) + 1)
    else:
        cellnorm = E.sum(axis=1, keepdims=True)
        cellnorm[cellnorm == 0] = 1 # 0/1 instead of 0/0 for cells with no dextramer signal
        C = E / cellnorm

    # clone purity
    clonal_counts = pd.DataFrame(E > 0).groupby(c).sum()
    total = clonal_counts.sum(axis=1)
    R = clonal_counts.div(total, axis=0).fillna(0)

    if faithful:
        non_zero = (clonal_counts != 0).astype(int)
        pure = non_zero.sum(axis=1) == 1
        R[pure] = non_zero[pure].div(total[pure], axis=0).fillna(0)
    
    R = R.loc[c].values
    
    # Dextramer signal correction (rows that summed 0 remain as 0)
    S = np.log(E+0.01) * R * C**2
    S[S<1] = 0

    # Per cell normalization: pMHC-wise log-ratio normalization
    cellnorm = S.sum(axis=1, keepdims=True)
    cellnorm[cellnorm == 0] = 1
    S = S / cellnorm
    
    # Dextramer normalization: cell-wise z-score normalization
    S = (S - S.mean(axis=0, keepdims=True)) / S.std(axis=0, ddof=1, keepdims=True)
    S[np.isnan(S)] = np.nanmin(S) # set NA's to smalles observed value

    assignment = (S > threshold).astype("uint8")
    if inplace:
        if is_mudata:
            adata.mod[dex_key].obsm["icon_pMHC_assignment"] = assignment
        else:
            adata.obsm["icon_pMHC_assignment"] = assignment
    else:
        return assignment
