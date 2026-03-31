from __future__ import annotations

from typing import List, Union

from scipy import stats
import numpy as np
import pandas as pd

import anndata as ad
import mudata as md

# Ignore small sample warning from scipy when calculating expected target for clonotypes with few cells
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*sample arguments is too small.*"
)

class ITRAP:
    """
    This class implements the ITRAP algorithm introduced by Povlsen et al. (2023).
    First each clonotype with more than 10 cells is assigned an expected target if the highest UMI count is
    significantly higher than the second most abundant pMHC using Wilcoxon p < 0.05.
    Each cell's specificity is then assigned to the most abundant pMHC based on UMI count.
    Using this expected target per clonotype, ITRAP calculates ideal UMI thresholds using a grid-search by optimizing
    the accuracy (if the epitope with highest UMI count of a cell matches the expected target) while preserving the
    ratio of retained cells using a weighted average between both objectives.
    The optimal thresholds are then used to filter cells. Further filtering steps may be
    included if the respective data, e.g., donor HLA, is available.
    """
    __name = "ITRAP"
    __version = "0.0.1"

    def __init__(self, filters=None):
        """
        Args:
            filters: List of filters to apply, options=['opt_thr', 'hashing_singlets', 'matching_HLA', 'complete_TCRs',
            'specificity_multiplets', 'is_cell'] (default: ['opt_thr'])
        """
        super().__init__()
        self.opt_thr = None
        self.filters = filters if filters is not None else ['opt_thr']
        self.data = None
        self.ir_clone_key = None
        self.specificity_to_idx = None
        self.idx_to_specificity = None

    def preprocess_model_data(
            self, 
            adata: Union[md.MuData, ad.AnnData], 
            pmhc_keys: Union[str, List[str]] = None, 
            neg_ctrl_key: str = None,
            ir_clone_key: str = 'clone_id',
            dex_key: str = "dex", 
            ir_key: str = "airr",
            umi_cols_TRA: list=None, umi_cols_TRB: list=None,
            is_cell_key: str = 'is_cell',
            chain_pairing_key: str = 'chain_pairing',
            hashing_classification_key: str = 'HTO_classification',
            **kwargs
        ):
        """
        Args:
            adata: A MuData object containing only dextramer counts and clonotype information,
                or an AnnData object containing the dextramer counts and clonotype information in the specified obsm and obs keys.
            pmhc_keys (Optional): A string or list of strings indicating the pMHC columns in `dex_key` modality which should be deconvolved.
                If None is given, the full dextramer matrix is used, excluding the negative control.
            neg_ctrl_key: A string specifying the negative control column in the `dex_key` matrix.
            ir_clone_key: A string specifying the field in `obs` that holds clonotype ids. If adata is a MuData object, this will be prefixed with `{ir_key}:`
            dex_key: the dextramer signal MuData module key, or the obsm key if adata is an AnnData object
            ir_key: the MuData module key where the immune receptor data is stored, only relevant if adata is a MuData object.
            umi_cols_TRA: list of strings specifying the columns in `obs` that hold the UMI counts for TRA, if available. If adata is a MuData object, these will be prefixed with `{ir_key}:`
            umi_cols_TRB: list of strings specifying the columns in `obs` that hold the UMI counts for TRB, if available. If adata is a MuData object, these will be prefixed with `{ir_key}:`
            is_cell_key: string specifying the column in `obs` that indicates whether a barcode is classified as a cell, only relevant if 'is_cell' filter is applied.
            chain_pairing_key: string specifying the column in `obs` that indicates whether a cell has complete TCR chain pairing, only relevant if 'complete_TCRs' filter is applied.
            hashing_classification_key: string specifying the column in `obs` that indicates the hashing classification of a cell, only relevant if 'hashing_singlets' filter is applied.
        """
        def calc_delta(x):
            """ Calculate UMI ratio of two most abundant pMHCs, 0.25 is a small constant to avoid division by zero"""
            if len(x) == 1:
                return x[-1] / 0.25
            elif len(x) == 0:
                return 0
            else:
                return (x.nlargest(2).iloc()[0]) / (x.nlargest(2).iloc()[1] + 0.25)
        
        # Check inputs
        if ir_clone_key is None:
            raise ValueError(f"{self.__name} requires a clonotype definition. Please specify a `ir_clone_key`.")
        if neg_ctrl_key is None:
            raise ValueError("No negative control specified. Please provide a `neg_ctrl_key` ")

        # Adjust data access for mudata and anndata
        if isinstance(adata, md.MuData):
            dex = adata.mod[dex_key]
            dex = pd.DataFrame(dex.X.toarray(), index=dex.obs_names, columns=dex.var_names)
            ir_clone_key = f'{ir_key}:{ir_clone_key}' if not ir_clone_key in adata.obs.columns else ir_clone_key
            chain_pairing_key = f'{ir_key}:{chain_pairing_key}' if not chain_pairing_key in adata.obs.columns else chain_pairing_key
            umi_cols_TRA = [f'{ir_key}:{col}' if not col in adata.obs.columns else col for col in umi_cols_TRA] if umi_cols_TRA is not None else None
            umi_cols_TRB = [f'{ir_key}:{col}' if not col in adata.obs.columns else col for col in umi_cols_TRB] if umi_cols_TRB is not None else None
            adata.pull_obs() # make sure adata.obs is updated with prefixed columns from ir module
            
        elif isinstance(adata, ad.AnnData):
            dex = adata.obsm[dex_key]
        
        if pmhc_keys is None:
            pmhc_keys = dex.columns[dex.columns != neg_ctrl_key].tolist()

        self.umi_cols_TRA = umi_cols_TRA
        self.umi_cols_TRB = umi_cols_TRB
        self.umi_cols_mhc =  [neg_ctrl_key] + pmhc_keys if type(pmhc_keys) == list else [neg_ctrl_key, pmhc_keys]

        # get dextramer counts
        data = dex.loc[:, self.umi_cols_mhc]
        self.specificity_to_idx = {s: i for i, s in enumerate(self.umi_cols_mhc)}
        self.idx_to_specificity = {i: s for i, s in enumerate(self.umi_cols_mhc)}

        # Get clonotype information and filters
        self.ir_clone_key = ir_clone_key
        self.is_cell_key = is_cell_key if 'is_cell' in self.filters else None
        self.chain_pairing_key = chain_pairing_key if 'complete_TCRs' in self.filters else None
        self.hashing_classification_key = hashing_classification_key if 'hashing_singlets' in self.filters else None
        for col in [self.ir_clone_key, self.is_cell_key, self.chain_pairing_key, self.hashing_classification_key]:
            if col is not None:
                if not col in adata.obs.columns:
                    raise ValueError(f"Filter {col} specified but column not found in adata.obs.")
                data[col] = adata.obs[col].values

        # Calculate UMI count and delta for pMHCs, TRA and TRB. Nomenclature follows original implementation
        # umi_count_X = max(UMI count of X)
        # delta_umi_X = ratio between highest and second highest UMI counts
        data['umi_count_mhc'] = data[self.umi_cols_mhc].max(1)
        data['delta_umi_mhc'] = data[self.umi_cols_mhc].apply(calc_delta, axis=1)
        data['umi_count_mhc_rel'] = data['umi_count_mhc'] / data['umi_count_mhc'].quantile(0.9, interpolation='lower')
        if umi_cols_TRA is not None:
            data['umi_count_TRA'] = adata.obs[umi_cols_TRA].max(1) if len(umi_cols_TRA) > 1 else adata.obs[umi_cols_TRA].values
            data['delta_umi_TRA'] = adata.obs[umi_cols_TRA].apply(calc_delta, axis=1)
        if umi_cols_TRB is not None:
            data['umi_count_TRB'] = adata.obs[umi_cols_TRB].max(1) if len(umi_cols_TRB) > 1 else adata.obs[umi_cols_TRB].values
            data['delta_umi_TRB'] = adata.obs[umi_cols_TRB].apply(calc_delta, axis=1)
        self.data = data

    def fit(self):
        """
        Fit the ITRAP model to the data. Calculate the ideal UMI thresholds for filtering
        """
        if self.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        # Calculate ideal thresholds
        self.opt_thr = self._calculate_ideal_umi_thresholds(self.data)

    def assign_pmhc(
            self, adata=None,
            is_cell_keep_values: List=[True],
            chain_pairing_keep_values: List=['single pair', 'extra VDJ', 'extra VJ'],
            hashing_classification_keep_values: List=['singlet', 'Singlet'],
        ) -> np.array:
        """
        Returns the binder assignments based on the most abundant UMI count for each cell.
        To filter out noise, different filters are applied to the data.
        Args:
            adata: If provided, the pMHC assignment will be added to adata.obs['itrap_pMHC_assignment'] and adata.obsm['itrap_pMHC_assignment'].
            is_cell_keep_values: List of values in `is_cell_key` column that indicate a barcode is classified as a cell, only relevant if 'is_cell' filter is applied.
            chain_pairing_keep_values: List of values in `chain_pairing_key` column that indicate a cell has complete TCR, only relevant if 'complete_TCRs' filter is applied.
            hashing_classification_keep_values: List of values in `hashing_classification_key` column that indicate a cell is a singlet, only relevant if 'hashing_singlets' filter is applied.
        Returns:
            An assignment array with the class assignment decision.
            If adata is not none, the assignment will be added to adata.obsm['itrap_pMHC_assignment'].
        """
        if self.opt_thr is None:
            print("Model has not been fit yet. Finding optimal thresholds...")
            self.fit()

        # Assign cells to most abundant pMHC based on UMI count, then set assignment to 0 if it fails filters
        self.data['assignment'] = self.data[self.umi_cols_mhc].idxmax(1).values
        self.data['assignment'] = self.data['assignment'].map(self.specificity_to_idx)
        self.data['assignment_before_filtering'] = self.data['assignment'].copy()
        filters = self._generate_filters(self.data, is_cell_keep_values, chain_pairing_keep_values, hashing_classification_keep_values)
        self.data.loc[~filters, 'assignment'] = 0
        assignments = pd.Series(self.data['assignment'].values.astype(int)).map(self.idx_to_specificity).values

        if adata is not None:
            adata.obs['itrap_pMHC_assignment'] = assignments
            adata.obsm['itrap_pMHC_assignment'] = pd.get_dummies(assignments).astype(int).set_index(adata.obs_names)
        return assignments

    def _generate_filters(
            self, data, is_cell_keep_values, chain_pairing_keep_values, hashing_classification_keep_values,
        ):
        filters = pd.Series([True] * len(data), index=data.index)

        # Filter 1: UMI thresholds
        if 'opt_thr' in self.filters:
            for k, thr in self.opt_thr.items():
                if k in data.columns:
                    filters &= data[k] >= thr

        # TODO Other filters are not implemented yet, only makes sense once we have the respective data
        # Filter 2: Hashing singlets
        if 'hashing_singlets' in self.filters:
            filters &= data[self.hashing_classification_key].isin(hashing_classification_keep_values).values

        # Filter 3: Matching HLA
        if 'matching_HLA' in self.filters:
            raise NotImplementedError("Matching HLA filter is not implemented yet.")

        # Filter 4: Complete TCRs
        if 'complete_TCRs' in self.filters:
            filters &= data[self.chain_pairing_key].isin(chain_pairing_keep_values).values

        # Filter 5: Specificity multiplets
        if 'specificity_multiplets' in self.filters:
            multiplets = data.groupby([self.ir_clone_key, 'assignment_before_filtering'], observed=True).size() > 1
            filters &= data.set_index([self.ir_clone_key, 'assignment_before_filtering']).index.map(multiplets).values

        # Filter 6: Is cell (GEX/cellranger/TCR) - user defined
        if 'is_cell' in self.filters:
            filters &= data[self.is_cell_key].isin(is_cell_keep_values).values

        return filters

    def _calculate_expected_target(self, data):
        # Select two most abundant pMHC based on UMI count
        most_abundant_epitope = data[self.umi_cols_mhc].sum(0).nlargest(2).index

        w, p = stats.wilcoxon(data[most_abundant_epitope[0]].fillna(0) - data[most_abundant_epitope[1]].fillna(0),
                              alternative='greater')

        if p <= 0.05:
            return True, most_abundant_epitope[0]
        else:
            return False, most_abundant_epitope[0]

    def _calculate_ideal_umi_thresholds(self, data):
        # In case of a tie, in default params negative control is the first column and hence chosen as most abundant
        data['cell_specificity'] = data[self.umi_cols_mhc].idxmax(1).values

        # Calculate expected target for each clonotype
        ct_pep = data.groupby(self.ir_clone_key, observed=True).filter(lambda x: len(x) >= 10)
        ct_pep = ct_pep.groupby(self.ir_clone_key, observed=True).apply(self._calculate_expected_target).to_frame()
        ct_pep[['significant', 'expected_target']] = ct_pep[0].apply(pd.Series)
        ct_pep = ct_pep[ct_pep['significant']].drop(columns=0)

        # Add expected target of each clonotype to full data and filter out non-significant clonotype targets
        data['ct_pep'] = data[self.ir_clone_key].map(ct_pep['expected_target'])
        cells_with_ct_pep = data[data['ct_pep'].notna()].copy()
        cells_with_ct_pep['pep_match'] = cells_with_ct_pep['cell_specificity'] == cells_with_ct_pep['ct_pep']

        # Grid search for optimal UMI threshold, hparams extracted from ITRAP code
        if self.umi_cols_TRA is None:
            umi_count_TRA_l = [None]
            delta_umi_TRA_l = [None]
        else:
            umi_count_TRA_l = np.arange(0, data['umi_count_TRA'].quantile(0.4, interpolation='higher'))
            delta_umi_TRA_l = np.arange(0, 4)
        if self.umi_cols_TRB is None:
            umi_count_TRB_l = [None]
            delta_umi_TRB_l = [None]
        else:
            umi_count_TRB_l = np.arange(0, data['umi_count_TRB'].quantile(0.4, interpolation='higher'))
            delta_umi_TRB_l = np.arange(0, 4)

        umi_count_mhc_l = np.arange(1, data['umi_count_mhc'].quantile(0.5, interpolation='higher'))
        delta_umi_mhc_l = [0, 1, 2]  # hparam from itrap Snakefile
        umi_relat_mhc_l = [0]  # seems unused in original implementation

        table = pd.DataFrame(columns=['accuracy', 'ratio_retained_gems', 'umi_count_mhc', 'umi_relat_mhc_l',
                                      'delta_umi_mhc', 'umi_count_TRA', 'delta_umi_TRA', 'umi_count_TRB',
                                      'delta_umi_TRB',])

        n_total_gems = len(cells_with_ct_pep)

        i = -1
        for uca in umi_count_TRA_l:
            for dua in delta_umi_TRA_l:
                for ucb in umi_count_TRB_l:
                    for dub in delta_umi_TRB_l:
                        for ucm in umi_count_mhc_l:
                            for urm in umi_relat_mhc_l:
                                for dum in delta_umi_mhc_l:
                                    i += 1
                                    filter_bool = ((cells_with_ct_pep['umi_count_mhc'] >= ucm) &
                                                   (cells_with_ct_pep['delta_umi_mhc'] >= dum) &
                                                   (cells_with_ct_pep['umi_count_mhc_rel'] >= urm))

                                    if self.umi_cols_TRA is not None:
                                        filter_bool &= (cells_with_ct_pep['umi_count_TRA'] >= uca) & (
                                                cells_with_ct_pep['delta_umi_TRA'] >= dua)
                                    if self.umi_cols_TRB is not None:
                                        filter_bool &= (cells_with_ct_pep['umi_count_TRB'] >= ucb) & (
                                                cells_with_ct_pep['delta_umi_TRB'] >= dub)

                                    flt = cells_with_ct_pep[filter_bool].copy()

                                    n_gems = len(flt)
                                    n_mat = flt['pep_match'].sum()

                                    g_ratio = round(n_gems / n_total_gems, 3)
                                    acc = round(n_mat / n_gems, 3)

                                    table.loc[i] = (acc, g_ratio, ucm, urm, dum, uca, dua, ucb, dub,)

        table['mix_mean'] = (table['accuracy'] * 2 + table['ratio_retained_gems']) / 3
        optimal_thresholds = (table.sort_values(by=['mix_mean', 'accuracy', 'ratio_retained_gems', 'umi_count_mhc'],
                                                ascending=[True, True, True, False]))
        opt_thr = optimal_thresholds.iloc()[-1][['umi_count_mhc', 'delta_umi_mhc', 'umi_count_TRA',
                                                 'delta_umi_TRA', 'umi_count_TRB', 'delta_umi_TRB']]

        return opt_thr
