import unittest

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


if __name__ == '__main__':
    unittest.main()
