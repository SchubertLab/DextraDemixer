import unittest

import numpy as np

import scirpy as ir
import scanpy as sc
from mudata import MuData
import muon as mu



class TestSimulation(unittest.TestCase):

    def setUp(self):
        adata_tcr = ir.io.read_10x_vdj(
            "../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_5pv2_Multiplex_vdj_t_all_contig_annotations.csv")

        adata = sc.read_10x_h5(
            "../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein_5pv2_Multiplex_count_raw_feature_bc_matrix.h5",
            gex_only=False)
        adata.var_names_make_unique()
        mdata = MuData({"gex": adata, "airr": adata_tcr})
        ir.pp.index_chains(mdata)
        ir.tl.chain_qc(mdata)

        # filter TCRs only and Antigen barcodes only
        mdata = mdata[mdata.obs["airr:receptor_type"] == "TCR"]
        mdata = mdata[:, mdata.var["gex:feature_types"] == "Antigen Capture"]

        # minimal pMHC QC filtering
        sc.pp.filter_cells(mdata["gex"], min_genes=1)
        sc.pp.filter_genes(mdata["gex"], min_cells=10)

        mdata.update()

        mu.pp.filter_obs(mdata, "airr:chain_pairing", lambda x: ~np.isin(x, ["orphan VDJ", "orphan VJ"]))
        ir.pp.ir_dist(mdata)
        ir.tl.define_clonotypes(mdata, receptor_arms="all", dual_ir="primary_only")

        ir.pp.ir_dist(mdata, metric="alignment", sequence="aa", cutoff=250)
        ir.tl.define_clonotype_clusters(mdata, sequence="aa", metric="alignment", receptor_arms="all", dual_ir="any")
        ir.tl.clonotype_network(mdata, min_cells=3, sequence="aa", metric="alignment")

        self.mdata = mdata

    def test_estimating_params(self):
        pass

    def test_simulating_params(self):
        pass