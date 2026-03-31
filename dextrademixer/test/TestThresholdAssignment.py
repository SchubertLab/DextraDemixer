import unittest
import numpy as np
import jax.numpy as jnp

from dextrademixer.model import threshold_assign_pmhc
from dextrademixer.utils import DextramerSimulator


class TestThresholdAssignment(unittest.TestCase):
    def test_threshold_based_assignment(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat, 10)
        print(mdat.mod["airr"].obs.clone_id)
        print(mdat.mod["gex"].X)
        print(assignment)

    def test_threshold_based_assignment_inplace(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat, 10, neg_ctrl_key="neg_control", inplace=True)
        print(mdat.mod["gex"].X)
        print(mdat.mod["gex"].obsm["pMHC_assignment"])


    def test_threshold_based_assignment_total_normalization(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat, 5, neg_ctrl_key="neg_control",
                                           total_normalization=True,
                                           target_sum=10e6)
        print(mdat.mod["gex"].X)
        print(assignment)

    def test_threshold_based_assignment_z_score(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat, 0, neg_ctrl_key="neg_control",
                                           z_score_normalization=True)
        print(mdat.mod["gex"].X)
        print(assignment)

    def test_threshold_based_assignment_z_score_and_total(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat, 0, neg_ctrl_key="neg_control",
                                           total_normalization=True,
                                           z_score_normalization=True)
        print(mdat.mod["gex"].X)
        print(assignment)

    def test_threshold_based_assignment_relative_threshold(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=10,
                                                        nof_clones=3,
                                                        binding_ratio=0.5,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)
        assignment = threshold_assign_pmhc(mdat,
                                           threshold=0.5,
                                           z_score_normalization=True,
                                           threshold_type="relative")
        print(mdat.mod["gex"].X)
        print(assignment)



if __name__ == '__main__':
    unittest.main()
