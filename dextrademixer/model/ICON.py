from typing import List, Union
import numpy as np
import pandas as pd
import mudata as md



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
                   deconvolved. If None is given, the full X is used, excluding the negative control if specified.
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
        pmhc_keys = gex.var_names[gex.var_names != neg_ctrl_key]

    if bg_noise is None:
        bg_noise = gex[:, neg_ctrl_key].X.max() if neg_ctrl_key is not None else 10

    X = gex[:, pmhc_keys].X.toarray()
    c = air.obs[ir_clone_key].to_numpy().astype("int32")

    # substract background
    E = np.maximum(0, X - bg_noise)
    ge0 = E.sum(axis=1) > 0 # 0 mask

    # calc pMHC ratio per cell
    C = E.copy()
    C[ge0] = E[ge0] / E[ge0].sum(axis=1, keepdims=True)

    # clone purity
    R = pd.DataFrame(E > 0).groupby(c).sum()
    R = R.div(R.sum(axis=1), axis=0).fillna(0).loc[c].values
    
    # Dextramer signal correction (rows that summed 0 remain as 0)
    S = np.log(E+0.01) * R * C**2

    # Per cell normalization: pMHC-wise log-ratio normalization
    S[ge0] = S[ge0] / S[ge0].sum(axis=1, keepdims=True)
    
    # Dextramer normalization: cell-wise z-score normalization
    S = (S - S.mean(axis=0, keepdims=True)) / S.std(axis=0, keepdims=True)

    assignment = (S > threshold).astype("uint8")
    if inplace:
        mdata.mod[gex_key].obsm["icon_pMHC_assignment"] = assignment
    else:
        return assignment
