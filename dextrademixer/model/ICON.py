from typing import List, Union
import numpy as np
import pandas as pd
import mudata as md
import anndata as ad


class ICON:
    """
    This class implements the ICON assignment procedure introduced by Zhang et al. (2021).

    ICON deconvolves all pMHCs jointly and returns a cells x pMHC
    assignment matrix rather than per-cell probabilities.
    """
    __name = "ICON"
    __version = "0.0.1"

    def __init__(self,
                 threshold: float = 0,
                 bg_noise: float = None,
                 bg_noise_quantile: float = 0.975,
                 faithful: bool = False,
                 ):
        """
        Args:
            threshold: A relative threshold to determine dextramer-specificity
            bg_noise: (Optional) A value to substract from dextramer counts to account for background
                noise. If None is given, the bg_noise_quantile of the negative control column is used
                if specified, otherwise 10.
            bg_noise_quantile: the quantile of the negative control used when `bg_noise` is None
            faithful: boolean indicating whether to use the original ICON procedure (True) or a
                debuged version based on the paper description
        """
        super().__init__()
        self.threshold = threshold
        self.bg_noise = bg_noise
        self.bg_noise_quantile = bg_noise_quantile
        self.faithful = faithful

        self.data = None
        self.assignment = None

    def preprocess_model_data(self,
                              data: Union[md.MuData, ad.AnnData],
                              ir_clone_key: str,
                              neg_ctrl_key: str = None,
                              pmhc_keys: Union[str, List[str]] = None,
                              pmhc_modality_key: str = "dex",
                              **kwargs):
        """
        Pulls the dextramer counts and clonotype ids out of `data`.

        Args:
            data: A MuData object containing only dextramer counts and clonotype information, an
                AnnData object containing the dextramer counts and clonotype information in the
                specified obsm and obs keys, or a cells x features DataFrame holding both the counts
                and the clonotype column.
            ir_clone_key: A string specifying the field in `obs` that holds clonotype ids.
                If in the immune receptor modality of a mudata object, should be
                `ir_modality_key:clone_key`.
            neg_ctrl_key: (Optional) a string specifying the negative control column in the
                `pmhc_modality_key` matrix.
            pmhc_keys: (Optional) A string or list of strings indicating the pMHC columns in
                `pmhc_modality_key` modality which should be deconvolved. If None is given, the full
                matrix is used, excluding the negative control if specified.
            pmhc_modality_key: the dextramer signal MuData module key, or the obsm key if data is an
                AnnData object

        Raises:
            ValueError: if `ir_clone_key` holds NA values.
            TypeError: if `data` is of an unsupported type.
        """
        # get dextramer counts
        if isinstance(data, md.MuData):
            is_mudata = True
            dex = data.mod[pmhc_modality_key]
            dex = dex.to_df()  # works for sparse and dense X
        elif isinstance(data, ad.AnnData):
            is_mudata = False
            dex = data.obsm[pmhc_modality_key]
        elif isinstance(data, pd.DataFrame):
            is_mudata = False
            dex = data  # cells x features, counts and annotation in one frame
        else:
            raise TypeError(f"unsupported input type {type(data).__name__}, expected a MuData, an "
                            f"AnnData or a cells x features DataFrame")

        # check if clone key contains NA values
        obs = data if isinstance(data, pd.DataFrame) else data.obs
        if obs[ir_clone_key].isna().sum() > 0:
            raise ValueError(f"NA values found in clone key {ir_clone_key} of data.obs. ICON works only for cells with TCR information. Please filter the object.")
        c = obs[ir_clone_key].to_numpy().astype("int32")

        if pmhc_keys is None:
            drop = {neg_ctrl_key, ir_clone_key} if dex is obs else {neg_ctrl_key}
            X = dex.loc[:, ~dex.columns.isin(drop)].values
        else:
            X = dex.loc[:, np.atleast_1d(pmhc_keys)].values

        # get background noise
        bg_noise = self.bg_noise
        if bg_noise is None:
            bg_noise = np.quantile(dex.loc[:, neg_ctrl_key], q=self.bg_noise_quantile) if neg_ctrl_key is not None else 10

        self.data = {"X": X, "c": c, "bg_noise": bg_noise, "obj": data, "is_mudata": is_mudata,
                     "pmhc_modality_key": pmhc_modality_key}

    def fit_scores(self):
        """
        Scores the data prepared by `preprocess_model_data` and stores the resulting assignment.

        Low-level path: `fit` does both steps in one call. Takes no arguments -- `threshold`,
        `bg_noise`, `bg_noise_quantile` and `faithful` are set on the constructor.

        Raises:
            Exception: if `preprocess_model_data` has not been called yet.
        """
        if self.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        X, c, bg_noise = self.data["X"], self.data["c"], self.data["bg_noise"]

        # substract background
        E = np.maximum(0, X - bg_noise)

        # calc pMHC ratio per cell
        if self.faithful:
            # +1 in the denominator can have large effects
            C = E / (E.sum(axis=1, keepdims=True) + 1)
        else:
            cellnorm = E.sum(axis=1, keepdims=True)
            cellnorm[cellnorm == 0] = 1  # 0/1 instead of 0/0 for cells with no dextramer signal
            C = E / cellnorm

        # clone purity
        clonal_counts = pd.DataFrame(E > 0).groupby(c).sum()
        total = clonal_counts.sum(axis=1)
        R = clonal_counts.div(total, axis=0).fillna(0)

        if self.faithful:
            non_zero = (clonal_counts != 0).astype(int)
            pure = non_zero.sum(axis=1) == 1
            R[pure] = non_zero[pure].div(total[pure], axis=0).fillna(0)

        R = R.loc[c].values

        # Dextramer signal correction (rows that summed 0 remain as 0)
        S = np.log(E+0.01) * R * C**2
        S[S < 1] = 0

        # Per cell normalization: pMHC-wise log-ratio normalization
        cellnorm = S.sum(axis=1, keepdims=True)
        cellnorm[cellnorm == 0] = 1
        S = S / cellnorm

        # Dextramer normalization: cell-wise z-score normalization
        S = (S - S.mean(axis=0, keepdims=True)) / S.std(axis=0, ddof=1, keepdims=True)
        S[np.isnan(S)] = np.nanmin(S)  # set NA's to smalles observed value

        self.assignment = (S > self.threshold).astype("uint8")

    def fit(self,
            data: Union[md.MuData, ad.AnnData, pd.DataFrame],
            *,
            ir_clone_key: str = None,
            neg_ctrl_key: str = None,
            pmhc_keys: Union[str, List[str]] = None,
            pmhc_modality_key: str = "dex",
            ) -> "ICON":
        """
        Extracts the model data from `data` and scores it, i.e. `preprocess_model_data` followed by
        `fit_scores`.

        Args:
            data: the dextramer counts, as a MuData, an AnnData or a cells x features DataFrame
            ir_clone_key: A string specifying the field in `obs` that holds clonotype ids
            neg_ctrl_key: (Optional) the negative control column in the `pmhc_modality_key` matrix
            pmhc_keys: (Optional) the pMHC columns to deconvolve, None uses all but the negative
                control
            pmhc_modality_key: the dextramer signal MuData module key, or the obsm key if data is an
                AnnData object

        Returns:
            self, so that `predict` can be chained onto the call
        """
        self.preprocess_model_data(data, ir_clone_key=ir_clone_key, neg_ctrl_key=neg_ctrl_key,
                                   pmhc_keys=pmhc_keys, pmhc_modality_key=pmhc_modality_key)
        self.fit_scores()
        return self

    def predict_posterior_class(self, inplace: bool = False) -> np.array:
        """
        Returns the assignment computed by `fit`.

        ICON does not produce posterior probabilities, so there is nothing to threshold here; the
        call is kept for interface parity with the other methods.

        Args:
            inplace: boolean indicating whether assignment should be stored in `obsm`. Not
                available for DataFrame input, which has no `obsm`

        Returns:
            A cells x pMHC array of assignments, or None if `inplace` is set, in which case the
            data object is modified adding an obsm matrix at `pmhc_modality_key`

        Raises:
            RuntimeError: if the model has not been fit yet.
            TypeError: if `inplace` is set but the input was a DataFrame.
        """
        if self.assignment is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit`.")

        if inplace:
            data, pmhc_modality_key = self.data["obj"], self.data["pmhc_modality_key"]
            if isinstance(data, pd.DataFrame):
                raise TypeError("`inplace` needs a MuData or AnnData to write `obsm` to; a "
                                "DataFrame input has nowhere to store the assignment.")
            if self.data["is_mudata"]:
                data.mod[pmhc_modality_key].obsm["icon_pMHC_assignment"] = self.assignment
            else:
                data.obsm["icon_pMHC_assignment"] = self.assignment
        else:
            return self.assignment

    def predict(self, *args, **kwargs):
        """Alias for `predict_posterior_class`."""
        return self.predict_posterior_class(*args, **kwargs)
