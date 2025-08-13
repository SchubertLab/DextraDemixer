import unittest

import jax
import muon as mu
import arviz as az
import numpyro as npy

import jax.numpy as jnp
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from dextrademixer.model import DextraDemixer
from dextrademixer.utils import DextramerSimulator, dist_to_sim
from dextrademixer.utils.simulation import t_cell_simulation


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
        print(DextraDemixer.available_methods())

    def test_svi_model_H(self):
        sim = DextramerSimulator()
        mdat, axis = sim.simulate_pmhc_data_from_distribution(total_cells=1000,
                                                              nof_clones=10,
                                                              simulate_neg_control=False,
                                                              use_clonotype_cov=True,
                                                              binding_fold_increase_range=[100],
                                                              variance_fold_increase_range=[1.2],
                                                              plot_data=True)

        binder = mdat.mod["airr"].obs["is_binder"]

        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "pmhc1")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_svi_model_I(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=1000,
                                                        nof_clones=10,
                                                        simulate_neg_control=False,
                                                        use_clonotype_cov=True,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)

        binder = mdat.mod["airr"].obs["is_binder"]

        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(mdat, "pmhc1")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_svi_model_C(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=100,
                                                        nof_clones=10,
                                                        simulate_neg_control=False,
                                                        use_clonotype_cov=True,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)

        binder = mdat.mod["airr"].obs["is_binder"]

        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5, clonotype_adherence=True)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_svi_model_C_C(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=1000,
                                                        nof_clones=10,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=True,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)

        binder = mdat.mod["airr"].obs["is_binder"]

        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_svi_model_C_neg_control(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=1000,
                                                        nof_clones=10,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=True,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False)

        binder = mdat.mod["airr"].obs["is_binder"]

        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(mdat, "pmhc1", ir_clone_key="clone_id", neg_ctrl_key="neg_control")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_svi_model_C_ir_cov(self):
        sim = DextramerSimulator()
        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=5000,
                                                        nof_clones=100,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=True,
                                                        p_binding_outlier=0.01,
                                                        binding_fold_increase_range=[100],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False, rng_key=4554)

        binder = mdat.mod["airr"].obs["is_binder"]
        c_nof = mdat.mod["airr"].uns["clone_cov"].shape[0]
        mdat.mod["airr"].uns["clone_cov"] = jnp.eye(c_nof)
        plt.show()

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "pmhc1",
                                    ir_cov_key="clone_cov",
                                    ir_clone_key="clone_id")
        trace = mixer.fit_svi(guide=npy.infer.autoguide.AutoNormal)  # AutoDelta works others dont
        print()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    @unittest.SkipTest
    def test_GPU_Metal(self):
        npy.set_platform("METAL")
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", size_factor_key="size_factor")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_H(self):
        npy.set_platform("cpu")
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
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
                                                         simulate_neg_control=True,
                                                         p_binding_outlier=0.01,
                                                         binding_fold_increase_range=[100],
                                                         variance_fold_increase_range=[1.2],
                                                         plot_data=False)

        binder = mdata.mod["airr"].obs["is_binder"]
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.05)
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_sampler_config_override(self):
        npy.set_platform("cpu")
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit(sampler_config={
            "num_samples": 500,
            "num_chains": 4,
            "progress_bar": True,
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
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_ppc_argmax(self):
        npy.set_platform("cpu")
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class()
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_ppc_fdr(self):
        npy.set_platform("cpu")
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        target_fdr = 0.1
        p, assignment = mixer.predict_posterior_class(target_fdr=target_fdr)
        N = len(self.binder)

        tn, fp, fn, tp = confusion_matrix(self.binder, assignment).ravel()
        tpr = tp / (tp + fn)
        fdr = fp / (tp + fp)

        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy, "FDR", fdr, tpr)
        self.assertAlmostEquals(target_fdr, ((fdr * 10 ** 2) // 1) / (10 ** 2))

    def test_simple_mixture_model_I(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)

        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_H(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_I(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_I(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p_c, assignment_c = mixer.predict_posterior_class(threshold=0.5, clonotype_adherence=True)
        p, assignment = mixer.predict_posterior_class(threshold=0.5, clonotype_adherence=False)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        accuracy_c = (self.binder == assignment_c).sum() / N
        print(list(self.binder))
        print(assignment.tolist())
        print(assignment_c.tolist())
        print(p.tolist())
        print(p_c.tolist())
        print("Accuracy", accuracy, accuracy_c)

    def test_simple_mixture_model_C_H(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_C(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_H(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_I(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_C(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_H(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_I(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_C(self):
        mixer = DextraDemixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.mdata, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id", ir_cov_key="ir_cov")
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_ture_data_cov(self):
        mdat = mu.read("../../data/10k_BEAM-T_Human_A0201_CMV_Flu_Covid_spikein.h5mu")

        mdat.mod["airr"].uns["clone_cov"] = dist_to_sim(mdat.mod["airr"].uns["ir_dist_aa_full"])
        #print(mdat)

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "CMV", neg_ctrl_key="negative_control",
                                    ir_cov_key="clone_cov",
                                    ir_clone_key="clone_id")
        trace = mixer.fit(sampler_config={"nuts": {"dense_mass": True}})
        print(mixer.summary())
        #print(mdat.mod["airr"].obs.is_binder)
        return True

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(mdat.mod["airr"].obs.is_binder)
        accuracy = (mdat.mod["airr"].obs.is_binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simulated_data_cov(self):
        sim = DextramerSimulator()
        mdat, axis = sim.simulate_pmhc_data_from_distribution(total_cells=3000,
                                                              nof_clones=100,
                                                              simulate_neg_control=True,
                                                              use_clonotype_cov=True,
                                                              binding_fold_increase_range=[100],
                                                              variance_fold_increase_range=[1.2],
                                                              plot_data=True)

        plt.show()
        binder = mdat.mod["airr"].obs["is_binder"]

        mixer = DextraDemixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(mdat, "pmhc1", neg_ctrl_key="neg_control",
                                    ir_cov_key="clone_cov",
                                    ir_clone_key="clone_id")
        trace = mixer.fit()
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class()
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        print(list(binder))
        print(assignment.tolist())
        print(p.tolist())
        print("Accuracy", accuracy)

    def test_kmean_initialization(self):

        sim = DextramerSimulator()

        mdat = sim.simulate_pmhc_data_from_distribution(total_cells=5000,
                                                        nof_clones=100,
                                                        binding_ratio=0.1,
                                                        simulate_neg_control=True,
                                                        use_clonotype_cov=False,
                                                        binding_fold_increase_range=[2],
                                                        variance_fold_increase_range=[1.2],
                                                        plot_data=False,
                                                        rng_key=756204)


        binder = mdat.mod["airr"].obs["is_binder"]

        mixer = DextraDemixer(model_type="mixturemodelkmeans", mode="I")

        mixer.preprocess_model_data(mdat,
                                    "pmhc1",
                                    #ir_cov_key="clone_cov",
                                    neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id"
                                    )

        trace = mixer.fit_svi(rng_key=1)
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class()
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
       # print(list(binder))
        #print(assignment.tolist())
        #print(p.tolist())
        print("Accuracy", accuracy)

        print("Random initialization")

        mixer = DextraDemixer(model_type="mixturemodelkmeans", mode="I")

        mixer.preprocess_model_data(mdat,
                                    "pmhc1",
                                    #ir_cov_key="clone_cov",
                                    neg_ctrl_key="neg_control",
                                    ir_clone_key="clone_id"
                                    )

        trace = mixer.fit_svi(use_minimal_loss=False, rng_key=1)
        print(mixer.summary())

        p, assignment = mixer.predict_posterior_class()
        N = len(binder)
        accuracy = (binder == assignment).sum() / N
        #print(list(binder))
        #print(assignment.tolist())
        #print(p.tolist())
        print("Accuracy", accuracy)


if __name__ == '__main__':
    unittest.main()
