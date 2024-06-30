import unittest

import numpy as np
import muon as mu
import pandas as pd

from dextramixer.model.BEAMT import BEAMT
from dextramixer.utils.simulation import t_cell_simulation


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.mdata = t_cell_simulation(n_clones=10,
                                       mean_binder_range=[350, 550],
                                       shape_binder_range=[300, 500],
                                       n_cells_per_binder=[50, 100],
                                       mean_non_binder=50,
                                       shape_non_binder=10,
                                       n_cells_per_non_binder=[50, 100],
                                       binding_ratio=0.5,
                                       rng_key=443)

        self.binder = self.mdata.mod["airr"].obs["is_binder"].to_numpy()
        self.mdata = mu.read("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein.h5mu")


    def test_BEAMT_fdr(self):
        beamt = BEAMT()
        beamt.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        beamt.fit()
        p, assignment = beamt.predict_posterior_class(target_fdr=0.05)
        print(assignment)
        print(self.binder)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_BEAMT_threshold(self):
        beamt = BEAMT()
        beamt.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        beamt.fit()
        p, assignment = beamt.predict_posterior_class(threshold=0.5)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_10x_genomics(self):
        df = pd.read_csv("../../data/antigen_specificity_scores.csv")
        df = df.set_index("barcode")

        mdat = self.mdata[:, ["CMV","negative_control"]]
        spec_score = df[df.antigen == "CMV"].loc[self.mdata.obs.index, "antigen_specificity_score"].to_numpy()

        beam = BEAMT()
        beam.preprocess_model_data(mdat, pmhc_key="CMV", neg_ctrl_key="negative_control")
        beam.fit()
        p, assignment = beam.predict_posterior_class(threshold=0.5)

        diverged = ~np.isclose(spec_score, np.round(p*100, decimals=3))
        print(spec_score[diverged])
        print(np.round(p[diverged]*100, decimals=4))
        #assert np.allclose(spec_score, np.round(p*100, decimals=4))


if __name__ == '__main__':
    unittest.main()
