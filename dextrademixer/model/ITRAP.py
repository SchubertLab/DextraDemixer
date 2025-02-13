from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Tuple

import mudata as md
import pandas as pd

import numpy as np
import jax.lax
import jax
from scipy import stats

from dextrademixer.model import ApMHCDeconvolution

if TYPE_CHECKING:
    from jax._src.typing import Array


class ITRAP(ApMHCDeconvolution):
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

    def __init__(self, umi_cols=None, umi_count_TRA=None, umi_count_TRB=None, filters=None):
        """
        Args:
            umi_cols: List of columns containing UMI counts for pMHCs (default set to ['neg_control', 'pmhc1'])
            umi_count_TRA: List of columns containing UMI counts for TRA (default: None)
            umi_count_TRB: List of columns containing UMI counts for TRB (default: None)
            filters: List of filters to apply, options=['opt_thr', 'hashing_singlets', 'matching_HLA', 'complete_TCRs',
            'specificity_multiplets', 'is_cell', 'viable_cells'] (default: ['opt_thr'])
        """
        super().__init__()
        self.opt_thr = None
        self.umi_cols_mhc = umi_cols
        self.umi_cols_TRA = umi_count_TRA
        self.umi_cols_TRB = umi_count_TRB
        self.filters = filters if filters is not None else ['opt_thr']
        self.data = None
        self.ir_clone_key = None
        self.specificity_to_idx = None
        self.idx_to_specificity = None

    def preprocess_model_data(self, mdata: md.MuData, pmhc_key: str, gex_key: str = "gex", neg_ctrl_key: str = None,
                              ir_key: str = "airr", ir_clone_key: str = None, ir_cov_key: str = None, **kwargs):
        if ir_clone_key is None:
            raise ValueError(f"{self.__name} requires a clonotype definition. Please specify a `ir_clone_key`.")

        gex = mdata.mod[gex_key]
        N = gex.shape[0]

        x = gex[:, pmhc_key].X.toarray().reshape((N,))
        x_neg = gex[:, neg_ctrl_key].X.toarray().reshape((N,))

        self._check_parameters(x, x_neg, None, None)
        self.ir_clone_key = ir_clone_key

        if self.umi_cols_mhc is None:
            if neg_ctrl_key is None:
                raise ValueError("No negative control specified and no umi_cols_mhc. Please provide a `neg_ctrl_key` "
                                 "or set umi_cols_mhc during initialization.")
            self.umi_cols_mhc = [neg_ctrl_key, pmhc_key]
        self.specificity_to_idx = {s: i for i, s in enumerate(self.umi_cols_mhc)}
        self.idx_to_specificity = {i: s for i, s in enumerate(self.umi_cols_mhc)}

        data = mdata['airr'].obs.copy()
        for col in self.umi_cols_mhc:
            data[col] = mdata['gex'][:, col].X.toarray().reshape(-1)

        def calc_delta(x):
            """ Calculate UMI ratio of two most abundant pMHCs, 0.25 is a small constant to avoid division by zero"""
            if len(x) == 1:
                return x[-1] / 0.25
            elif len(x) == 0:
                return 0
            else:
                return (x.nlargest(2).iloc()[0]) / (x.nlargest(2).iloc()[1] + 0.25)

        # Calculate UMI count and delta for pMHCs, TRA and TRB. Nomenclature follows original implementation
        # umi_count_X = max(UMI count of X)
        # delta_umi_X = ratio between highest and second highest UMI counts
        data['umi_count_mhc'] = data[self.umi_cols_mhc].max(1)
        data['delta_umi_mhc'] = data[self.umi_cols_mhc].apply(calc_delta, axis=1)
        data['umi_count_mhc_rel'] = data['umi_count_mhc'] / data['umi_count_mhc'].quantile(0.9, interpolation='lower')
        if self.umi_cols_TRA is not None:
            data['umi_count_TRA'] = data[self.umi_cols_TRA].max(1)
            data['delta_umi_TRA'] = data[self.umi_cols_TRA].apply(calc_delta)
        if self.umi_cols_TRB is not None:
            data['umi_count_TRB'] = data[self.umi_cols_TRA].max(1)
            data['delta_umi_TRB'] = data[self.umi_cols_TRB].apply(calc_delta)

        self.data = data

    def fit(self):
        """
        Fit the ITRAP model to the data. Calculate the ideal UMI thresholds for filtering
        """
        if self.data is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        # Calculate ideal thresholds
        self.opt_thr = self._calculate_ideal_umi_thresholds(self.data)

    def predict_posterior_class(self, threshold: float = None, target_fdr: float = None) -> Tuple[Array, Array]:
        """
        Returns the binder assignments based on the most abundant UMI count for each cell.
        To filter out noise, different filters are applied to the data.
        ITRAP does not return a posterior probability, so the assignment is returned as pseudo value.
        Threshold and target_fdr are ignored in this implementation.

        Args:
             threshold: (Optional) ignored
             target_fdr: (Optional) ignored
        Returns:
            A tuple (p, assignment) of arrays with p being the pseudo value for compatibility of binding and assignment
            the class assignment decision
        """
        if self.opt_thr is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit`.")

        # Assign cells to most abundant pMHC based on UMI count, then set assignment to 0 if it fails filters
        filters = self._generate_filters(self.data)
        self.data['assignment'] = self.data[self.umi_cols_mhc].idxmax(1).values
        self.data['assignment'] = self.data['assignment'].map(self.specificity_to_idx)
        self.data['assignment_before_filtering'] = self.data['assignment'].copy()
        self.data.loc[~filters, 'assignment'] = 0

        return self.data['assignment'].values.astype(int), self.data['assignment'].values.astype(float)

    def _generate_filters(self, data):
        filters = pd.Series([True] * len(data), index=data.index)

        # Filter 1: UMI thresholds
        if 'opt_thr' in self.filters:
            for k, thr in self.opt_thr.items():
                if k in data.columns:
                    filters &= data[k] >= thr
            # filters &= eval(' & '.join([f'(data["{k}"] >= {abs(v)})' for k, v in self.opt_thr.items() if k in data.columns]))

        # TODO Other filters are not implemented yet
        # Filter 2: Hashing singlets
        if 'hashing_singlets' in self.filters:
            raise NotImplementedError("Hashing singlets filter is not implemented yet.")

        # Filter 3: Matching HLA
        if 'matching_HLA' in self.filters:
            raise NotImplementedError("Matching HLA filter is not implemented yet.")

        # Filter 4: Complete TCRs
        if 'complete_TCRs' in self.filters:
            raise NotImplementedError("Complete TCRs filter is not implemented yet.")

        # Filter 5: Specificity multiplets
        if 'specificity_multiplets' in self.filters:
            raise NotImplementedError("Specificity multiplets filter is not implemented yet.")

        # Filter 6: Is cell (Cellranger)
        if 'is_cell' in self.filters:
            raise NotImplementedError("Is cell filter is not implemented yet.")

        # Filter 7: Viable cells (GEX)
        if 'viable_cells' in self.filters:
            raise NotImplementedError("Viable cells filter is not implemented yet.")

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
        # TODO What if tie in specificity? Should we just take the first one?
        data['cell_specificity'] = data[self.umi_cols_mhc].idxmax(1).values

        # Calculate expected target for each clonotype
        ct_pep = data.groupby(self.ir_clone_key).filter(lambda x: len(x) >= 10)
        ct_pep = ct_pep.groupby(self.ir_clone_key).apply(self._calculate_expected_target).to_frame()
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
        # TODO Why start from 1 in original implementation?
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


if __name__ == "__main__":
    from dextrademixer.utils import DextramerSimulator
    import muon as mu


    if os.path.exists('../../data/test.h5mu'):
        mdata = mu.read('../../data/test.h5mu')
    else:

        sim = DextramerSimulator()
        mdata = sim.simulate_pmhc_data_from_distribution(total_cells=10000,
                                                        nof_clones=150,
                                                        p_binding_outlier=0.05,
                                                        binding_ratio=0.1,
                                                        binding_fold_increase_range=[5],
                                                        variance_fold_increase_range=[1.2],
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=True,
                                                        plot_data=False,
                                                        rng_key=42)
        mdata.write('../../mdata/test.h5mu')

    itrap = ITRAP(filters=['opt_thr'])
    itrap.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control", ir_key="airr", ir_clone_key='clone_id')
    itrap.fit()
    p, assignment = itrap.predict_posterior_class()
    print(assignment)


from dextrademixer.utils.simulation import DextramerSimulator

sim = DextramerSimulator()
mdata = sim.simulate_pmhc_data_from_distribution(total_cells=500, nof_clones=10, binding_ratio=0.05,
                                                      simulate_neg_control=True, rng_key=42
                                                      )



binder = mdata.mod["airr"].obs["is_binder"].to_numpy()

itrap = ITRAP()
itrap.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control", ir_clone_key="clone_id")
itrap.fit()
p, assignment = itrap.predict_posterior_class(target_fdr=0.05)
print(assignment)
print(binder)
N = len(binder)
accuracy = (binder == assignment).sum() / N
print("Accuracy", accuracy)
