import unittest
import muon as mu

from dextramixer.utils import *


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = jnp.array([[0, 2, 3],
                               [2, 0, 4],
                               [3, 4, 0]])
        self.non_psd_matrix = jnp.array([[1, -2], [-2, 1]])

    def test_dist_to_cov(self):
        cov = dist_to_cov_psd(self.dist)
        eig = jnp.linalg.eigvals(cov)
        assert jnp.logical_or(eig > 0,  jnp.isclose(eig, 0)).all()

    def test_near_PSD(self):
        cov = nearest_psd(self.non_psd_matrix)
        assert (cov.T == cov).all()
        eig = jnp.linalg.eigvals(cov)
        assert jnp.logical_or(eig > 0,  jnp.isclose(eig, 0)).all()



if __name__ == '__main__':
    unittest.main()
