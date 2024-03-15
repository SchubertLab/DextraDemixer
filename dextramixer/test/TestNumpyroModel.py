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
        self.C = self.df_data.clone.to_xarray()
        self.binder = self.df_data.binder.to_xarray()
        self.Sigma = np.eye(len(pd.unique(self.C)))
        self.neg_cont = self.df_neg_cont.avidity.to_xarray()

    def test_model_registration(self):
       print(DextraMixer.available_methods())

    def test_simple_mixture_model(self):
        mixer = DextraMixer(model_type="mixturemodel")
        mixer.preprocess_model_data(self.X)
        trace = mixer.fit()
        print(mixer.sampler.print_summary())

        #z = mixer.predict_posterior_class(trace)
        #idx = z.mean(("chain", "draw"))
        #N = len(self.binder)
        #accuracy = (self.binder == idx).sum() / N
        #print(accuracy)

    def test_simple_mixture_mode(self):
       pass

    def test_simple_mixture_neg_control(self):
       pass

    def test_simple_mixture_model_C(self):
        pass

    def test_simple_mixture_model_Sigma(self):
        pass


if __name__ == '__main__':
    unittest.main()
