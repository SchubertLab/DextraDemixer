import unittest
import pandas as pd
import numpy as np
import arviz as az
import numpyro as npy

import jax.numpy as jnp

from dextramixer.model import DextraMixer
from dextramixer.utils import t_cell_simulation


class MyTestCase(unittest.TestCase):

    def setUp(self):
        npy.set_platform("cpu")
        npy.set_host_device_count(7)

        self.df_data, self.df_neg_cont = t_cell_simulation(n_clones=10,
                                                           mean_binder_range=[350,550],
                                                           shape_binder_range=[300,500],
                                                           n_cells_per_binder = [50,100],
                                                           mean_non_binder=50,
                                                           shape_non_binder=10,
                                                           n_cells_per_non_binder = [50,100],
                                                           binding_ratio=0.5,
                                                           rnd=43)

        self.X = jnp.array(self.df_data.avidity.to_xarray())
        self.C = jnp.array(self.df_data.clone.to_numpy())
        self.size_factor = jnp.ones(shape=(self.X.shape[0], 1))
        self.binder = jnp.array(self.df_data.binder.to_xarray())
        self.Sigma = jnp.eye(len(jnp.unique(self.C)))
        self.neg_cont = jnp.array(self.df_neg_cont.avidity.to_xarray())
        self.size_factor_neg = jnp.ones(self.neg_cont.shape[0])
        self.neg_c = jnp.array(self.df_neg_cont.clone.to_numpy())

    def test_model_registration(self):
       print(DextraMixer.available_methods())

    @unittest.SkipTest
    def test_GPU_Metal(self):
        npy.set_platform("METAL")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_H(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_ppc_threshold(self):
        npy.set_platform("cpu")
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(threshold=0.5)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, self.size_factor)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor,
                                    neg_x=self.neg_cont,
                                    size_factor_neg=self.size_factor_neg)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, self.size_factor,
                                    neg_x=self.neg_cont,
                                    size_factor_neg=self.size_factor_neg)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_H(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma,
                                    neg_x=self.neg_cont, size_factor_neg=self.size_factor_neg)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_I(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma,
                                    neg_x=self.neg_cont, size_factor_neg=self.size_factor_neg)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="C")
        mixer.preprocess_model_data(self.X, self.size_factor, c=self.C, sigma=self.Sigma,
                                    neg_x=self.neg_cont, size_factor_neg=self.size_factor_neg, c_neg=self.neg_c)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)


if __name__ == '__main__':
    unittest.main()
