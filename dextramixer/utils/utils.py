import jax.numpy as jnp


#Metaclass for Plugins
class RegisteredModel(type):
    def __new__(cls, clsname, superclasses, attributedict):
        newclass = type.__new__(cls, clsname, superclasses, attributedict)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()

        # condition to prevent base class registration
        if superclasses:
            cls.registry.setdefault(str(newclass().name).lower(), newclass)
        return newclass


def dist_to_cov(d):
    """
    converts a negative semi-definite matrix of squared differences to a positive semi-definite covariance matrix using
    the centering method of Gower (1996).

    see reference implementation: https://rdrr.io/cran/rwc/src/R/cov.from.dist.R

    Gower, J. C. (1966). Some distance properties of latent root and vector methods used in multivariate analysis.
    Biometrika, 53(3-4), 325-338.
    """
    if d.shape[0] != d.shape[1]:
        raise ValueError(f"Distance matrix must be squared but is {d.shape}")

    n = d.shape[0]
    one = jnp.ones(shape=(n, 1))
    sum_all = jnp.sum(d)/n**2
    row_sum_matrix = jnp.matmul(jnp.mean(d, axis=0), one.T)
    col_sum_matrix = jnp.matmul(one, jnp.mean(d, axis=1).T)
    return 1/2 * (-d + row_sum_matrix + col_sum_matrix - sum_all)


def sim_to_dist(s):
    """
    converts a quadratic similarity matrix into a distance matrix
    """
    if s.shape[0] != s.shape[1]:
        raise ValueError(f"Similarity matrix must be squared but is {s.shape}")

    EPS = jnp.finfo("float64").eps
    return - jnp.log((EPS + s) / (EPS + jnp.max(s)))