import unittest
import pandas as pd
import numpy as np
import arviz as az
import numpyro as npy

import jax.numpy as jnp
from matplotlib import pyplot as plt

from dextramixer.model import DextraMixer
from dextramixer.utils import DextramerSimulator
from dextramixer.utils.simulation import t_cell_simulation


class MyTestCase(unittest.TestCase):

    def setUp(self):
        npy.set_platform("cpu")
        npy.set_host_device_count(4)
        #self.mdata = DextramerSimulator().simulate_pmhc_data(total_cells=10, nof_clones=4, binding_ratio=0.5)
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
        #print(self.mdata)

    def test_model_registration(self):
        print(DextraMixer.available_methods())

    @unittest.SkipTest
    def test_GPU_Metal(self):
        npy.set_platform("METAL")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", size_factor_key="size_factor")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_H(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_H_simulation(self):
        npy.set_platform("cpu")
        sim = DextramerSimulator()
        mdata = sim.simulate_pmhc_data_from_distribution(total_cells=5000,
                                                         nof_clones=50,
                                                         binding_ratio=0.05,
                                                         simulate_neg_control=True,
                                                         binding_fold_increase_range=[500],
                                                         plot_data=False)

        binder = mdata.mod["airr"].obs["is_binder"]
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.05)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_sampler_config_override(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit(sampler_config={
            "num_samples": 500,
            "num_chains": 4,
            "progress_bar": False,
            "nuts": {
                "target_accept_prob": 0.95,
                "max_tree_depth": 10
            }
        })
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_ppc_threshold(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_ppc_argmax(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class()
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)

        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)


if __name__ == '__main__':
    unittest.main()
