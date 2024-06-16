import unittest

import numpy as np
import pandas as pd

import scirpy as ir
import scanpy as sc
from mudata import MuData
import muon as mu

from matplotlib import pyplot as plt

from dextramixer.utils import DextramerSimulator


class TestSimulation(unittest.TestCase):

    def setUp(self):
        # adata_tcr = ir.io.read_10x_vdj(
        #     "../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_5pv2_Multiplex_vdj_t_all_contig_annotations.csv")
        #
        # adata = sc.read_10x_h5(
        #     "../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_5pv2_Multiplex_count_raw_feature_bc_matrix.h5",
        #     gex_only=False)
        # adata.var_names_make_unique()
        # mdata = MuData({"gex": adata, "airr": adata_tcr})
        # ir.pp.index_chains(mdata)
        # ir.tl.chain_qc(mdata)
        #
        # # filter TCRs only and antigen barcodes only
        # mdata = mdata[mdata.obs["airr:receptor_type"] == "TCR"]
        # mdata = mdata[:, mdata.var["gex:feature_types"] == "Antigen Capture"]
        #
        # # minimal pMHC QC filtering
        # sc.pp.filter_cells(mdata["gex"], min_genes=1)
        # sc.pp.filter_genes(mdata["gex"], min_cells=10)
        #
        # mdata.update()
        #
        # mu.pp.filter_obs(mdata, "airr:chain_pairing", lambda x: ~np.isin(x, ["orphan VDJ", "orphan VJ"]))
        # ir.pp.ir_dist(mdata)
        # ir.tl.define_clonotypes(mdata, receptor_arms="all", dual_ir="primary_only")
        #
        # ir.pp.ir_dist(mdata, metric="alignment", sequence="aa", cutoff=250)
        # ir.tl.define_clonotype_clusters(mdata, sequence="aa", metric="alignment", receptor_arms="all", dual_ir="any")
        # ir.tl.clonotype_network(mdata, min_cells=3, sequence="aa", metric="alignment")
        #
        # self.mdata = mdata
        # dist = self.mdata.mod["airr"].uns["cc_aa_alignment"]["distances"].toarray()
        # self.mdata.mod["airr"].uns["ir_dist_aa_full"] = dist - 1
        # self.mdata.write("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein.h5mu")
        self.mdata = mu.read("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein.h5mu")
        #print(self.mdata)
        pass

    def test_estimating_params(self):
        sim = DextramerSimulator()
        sim.estimate_simulation_params(self.mdata, neg_ctrl_key="negative_control", ir_dist_key="ir_dist_aa_full")
        print(DextramerSimulator.default_params())
        print(sim.dist_params)
        #print(sim.params)

    def test_estimating_params_with_plot(self):
        from matplotlib import pyplot as plt

        sim = DextramerSimulator()
        ax = sim.estimate_simulation_params(self.mdata, neg_ctrl_key="negative_control",
                                            ir_dist_key="ir_dist_aa_full", plot_qc=True)
        print(sim.dist_params)
        plt.savefig("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_fitted_model.pdf")
        plt.show()

    def test_estimating_params_with_plot_filtered(self):
        from matplotlib import pyplot as plt

        sim = DextramerSimulator()
        ax = sim.estimate_simulation_params(self.mdata, neg_ctrl_key="negative_control",
                                            ir_dist_key="ir_dist_aa_full", filter_extreme_values=True, plot_qc=True)
        plt.savefig("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_fitted_model_filtered.pdf")
        plt.show()

    def test_estimating_params_with_plot_filtered_individually(self):
        sim = DextramerSimulator()
        ax = sim.estimate_simulation_params(self.mdata, neg_ctrl_key="negative_control",
                                            ir_dist_key="ir_dist_aa_full",
                                            filter_extreme_values=[True, True, False, False, True],
                                            plot_qc=True)
        print(sim.dist_params)
        plt.savefig("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_fitted_model_filtered.pdf")
        plt.show()

    def test_simulating_params(self):
        sim = DextramerSimulator()
        mdat, axs = sim.simulate_pmhc_data_from_distribution(total_cells=5000,
                                                             binding_ratio=0.1,
                                                             nof_clones=50,
                                                             binding_fold_increase_range=[10],
                                                             variance_fold_increase_range=[1.2],
                                                             simulate_neg_control=True,
                                                             plot_data=True)

        plt.show()

    def test_simulating_params_nctrl(self):
        sim = DextramerSimulator()
        mdat, _ = sim.simulate_pmhc_data_from_distribution(simulate_neg_control=True)
        print(mdat)

    def test_simulating_params_write_read(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(simulate_neg_control=True, use_clonotype_cov=True,
)

        mdat.write("test.h5mu")
        mdat2 = mu.read("test.h5mu")
        print(mdat2)

    def test_simulating_params_cov(self):
        sim = DextramerSimulator()
        mdat, _ = sim.simulate_pmhc_data_from_distribution(use_clonotype_cov=True)
        print(mdat)

    def test_simulation_sample(self):
        sim = DextramerSimulator()
        sim.estimate_simulation_params(self.mdata,
                                       neg_ctrl_key="negative_control",
                                       ir_dist_key="ir_dist_aa_full",
                                       #filter_extreme_values=[True, True, False, True]
                                       )
        mdat, axs = sim.simulate_pmhc_data_from_sample(total_cells=5000,
                                                       binding_ratio=0.1,
                                                       nof_clones=50,
                                                       binding_fold_increase_range=[500],
                                                       simulate_neg_control=True,
                                                       use_clonotype_cov=True,
                                                       plot_data=True)
        plt.show()
