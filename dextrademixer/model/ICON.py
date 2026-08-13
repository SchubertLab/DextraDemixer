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
                     inplace: bool = False,
                     faithful: bool = False,
                     ) -> np.ndarray | None:
    """
    Assign pMHC specificity with the ICON procedure.

    Parameters
    ----------
    adata : mudata.MuData or anndata.AnnData
        Object containing dextramer counts and cell-level clonotype IDs.
    ir_clone_key : str
        Column in ``adata.obs`` containing clonotype IDs. For MuData, a
        modality-prefixed key such as ``"airr:clone_id"`` may be used.
    neg_ctrl_key : str, optional
        Negative-control feature in the dextramer matrix.
    threshold : float, default=0
        Relative score threshold used to assign specificity.
    bg_noise : float, optional
        Background count subtracted from every dextramer feature. If omitted,
        it is estimated from ``neg_ctrl_key`` or defaults to 10.
    bg_noise_quantile : float, default=0.975
        Negative-control quantile used to estimate background noise.
    pmhc_keys : str or list of str, optional
        PMHC features to assign. By default, use all features except the
        negative control.
    dex_key : str, default="dex"
        MuData modality key or AnnData ``obsm`` key containing dextramer
        counts.
    inplace : bool, default=False
        Store assignments in ``obsm`` instead of returning them.
    faithful : bool, default=False
        Reproduce the original ICON implementation when true; otherwise use
        the corrected procedure described in the publication.

    Returns
    -------
    numpy.ndarray or None
        Binary cell-by-pMHC assignment matrix, or ``None`` when ``inplace`` is
        true.

    Raises
    ------
    ValueError
        If clonotype IDs contain missing values.
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
    else:
        X = dex.loc[:, np.atleast_1d(pmhc_keys)].values

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
