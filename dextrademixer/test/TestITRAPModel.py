import unittest

import numpy as np
import muon as mu
import pandas as pd

from dextrademixer.model.ITRAP import ITRAP
from dextrademixer.utils.simulation import DextramerSimulator


class MyTestCase(unittest.TestCase):

    def setUp(self):
        sim = DextramerSimulator()
        self.mdata = sim.simulate_pmhc_data_from_distribution(total_cells=500, nof_clones=10, binding_ratio=0.05,
                                                              simulate_neg_control=True, rng_key=42
                                                              )

        self.binder = self.mdata.mod["airr"].obs["is_binder"].to_numpy()

    def test_ITRAP(self):
        itrap = ITRAP()
        itrap.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control", ir_clone_key="clone_id")
        itrap.fit()
        p, assignment = itrap.predict_posterior_class(target_fdr=0.05)
        print(assignment)
        print(self.binder)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)



if __name__ == '__main__':
    unittest.main()
