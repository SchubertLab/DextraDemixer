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
        npy.set_host_device_count(4)
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
        self.binder = jnp.array(self.df_data.binder.to_xarray())
        self.Sigma = jnp.eye(len(jnp.unique(self.C)))
        self.neg_cont = jnp.array(self.df_neg_cont.avidity.to_xarray())

    def test_model_registration(self):
       print(DextraMixer.available_methods())

    def test_simple_mixture_model(self):
        mixer = DextraMixer(model_type="mixturemodel")
        mixer.preprocess_model_data(self.X)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_mode(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_neg_control(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, neg_cont=self.neg_cont)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_C(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="H")
        mixer.preprocess_model_data(self.X, c=self.C)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_Sigma(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, c=self.C, sigma=self.Sigma)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)

    def test_simple_mixture_model_full(self):
        mixer = DextraMixer(model_type="mixturemodel", mode="I")
        mixer.preprocess_model_data(self.X, c=self.C, sigma=self.Sigma, neg_cont=self.neg_cont)
        trace = mixer.fit()
        print(az.summary(trace, var_names=["~log_p"]))

        p, assignment = mixer.predict_posterior_class(target_fdr=0.001)
        N = len(self.binder)
        accuracy = (self.binder == assignment).sum() / N
        print("Accuracy", accuracy)


if __name__ == '__main__':
    unittest.main()
