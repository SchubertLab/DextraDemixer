import unittest

import numpy as np

from dextramixer.model.pymc_model import *
from dextramixer.utils import t_cell_simulation


class TestPymcModels(unittest.TestCase):

    def setUp(self):
        self.df_data, self.df_neg_cont = t_cell_simulation(n_clones=10,
                                         mean_binder_range=[200,550],
                                         shape_binder_range=[100,500],
                                         n_cells_per_binder=[50,100],
                                         mean_non_binder=50,
                                         shape_non_binder=10,
                                         n_cells_per_non_binder=[50,100],
                                         binding_ratio=0.5,
                                         rnd=434)
        self.X = self.df_data.avidity.values
        self.C = self.df_data.clone.values
        self.binder = self.df_data.binder.values
        self.Sigma = np.eye(len(pd.unique(self.C)))
        self.neg_cont = self.df_neg_cont.avidity.values

    def test_model_registration(self):
       print(DextraMixerPymc.available_methods())

    def test_simple_mixture_model(self):
        mixer = DextraMixerPymc()
        mixer.build_model(self.X)
        trace = mixer.fit()
        print(az.summary(trace))

        z = mixer.predict_posterior_class(trace)
        idx = z.mean(("chain", "draw"))
        N = len(self.binder)
        accuracy = (self.binder == idx).sum() / N
        print(accuracy)

    def test_simple_mixture_neg_control(self):
        mixer = DextraMixerPymc()
        mixer.build_model(self.X, negCont=self.neg_cont)
        trace = mixer.fit()
        print(az.summary(trace))

        z = mixer.predict_posterior_class(trace)
        idx = z.mean(("chain", "draw"))
        N = len(self.binder)
        accuracy = (self.binder == idx).sum() / N
        print(accuracy)

    def test_simple_mixture_model_C(self):
        mixer = DextraMixerPymc()
        mixer.build_model(self.df_data.avidity.values,  C=self.C)
        trace = mixer.fit()
        t = az.summary(trace)
        print(t)

        z = mixer.predict_posterior_class(trace)
        idx = z.mean(("chain", "draw"))
        N = len(self.binder)
        accuracy = (self.binder == idx).sum() / N
        print("PPC:", accuracy)
        idx = np.array([ 1 if t.loc['w[%i, 1]'%i, 'mean'] > 0.5 else 0 for i in self.C])
        print("Posterior class weight:", (self.binder == idx).sum() / N)

    def test_simple_mixture_model_Sigma(self):
        mixer = DextraMixerPymc()
        mixer.build_model(self.df_data.avidity.values,  C=self.C, Sigma=self.Sigma)
        trace = mixer.fit()
        t = az.summary(trace)
        print(t)

        z = mixer.predict_posterior_class(trace)
        idx = z.mean(("chain", "draw"))
        N = len(self.binder)
        accuracy = (self.binder == idx).sum() / N
        print("PPC:", accuracy)
        idx = np.array([ 1 if t.loc['w[%i, 1]'%i, 'mean'] > 0.5 else 0 for i in self.C])
        print("Posterior class weight:", (self.binder == idx).sum() / N)


if __name__ == '__main__':
    unittest.main()
