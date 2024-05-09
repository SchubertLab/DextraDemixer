from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy import ndarray, dtype, bool_
from scipy.stats import ortho_group, random_correlation


def dist_to_cov(d: jax.Array) -> jax.Array:
    """
    This computes a covariance matrix from a squared-distance matrix using the centering method of Gower (1996).

    see reference implementation: https://rdrr.io/cran/rwc/src/R/cov.from.dist.R

    Gower, J. C. (1966). Some distance properties of latent root and vector methods used in multivariate analysis.
    Biometrika, 53(3-4), 325-338.
    """
    if d.shape[0] != d.shape[1] or jnp.any(d != d.T):
        raise ValueError(f"Distance matrix must be square and symmetric.")

    n = d.shape[0]
    one = jnp.ones(shape=(n, 1))
    sum_all = jnp.sum(d) / n ** 2
    row_sum_matrix = jnp.matmul(jnp.mean(d, axis=0, keepdims=True).T, one.T)
    col_sum_matrix = jnp.matmul(one, jnp.mean(d, axis=1, keepdims=True).T)
    return 0.5 * (-d + row_sum_matrix + col_sum_matrix - sum_all)


def sim_to_dist(s: jax.Array) -> jax.Array:
    """
    converts a quadratic similarity matrix into a distance matrix
    """
    if s.shape[0] != s.shape[1] or jnp.any(s != s.T):
        raise ValueError(f"Similarity matrix must be square and symmetric.")

    EPS = jnp.finfo("float64").eps
    return - jnp.log((EPS + s) / (EPS + jnp.max(s)))


def sample_orthogonal_mtx(n: int, rng_key: int = 42) -> np.ndarray:
    """
    samples an orthogonal matrix of size nxn

    Args:
        n: dimension size of matrix
        rng_key: a random seed
    Returns:
        A nxn orthonormal matrix
    """
    return ortho_group.rvs(dim=n, random_state=rng_key)


def sample_cov_from_eigs(eigs: jax.Array, rng_key: int = 42) -> ndarray[Any, dtype[bool_]]:
    """
    samples a covariance matrix sampling an orthogonal matrix and multiplying it with eigenvalues
    Args:
        eigs: a list of eigenvalues of size n
        rng_key: a random seed
    Returns:
        a covariance matrix of size nxn
    """
    S = np.diag(eigs)
    Q = sample_orthogonal_mtx(eigs.shape[0], rng_key=rng_key)
    return Q.T @ S @ Q


def sample_corr_from_eigen(eigs: jax.Array, rng_key: int = 42) -> jax.Array:
    return random_correlation.rvs(dim=eigs, random_state=rng_key)


def remove_outliers(sr, iq_range=0.8):
    #  https://stackoverflow.com/a/39424972
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(sr[~np.isnan(sr)], [pcnt, 0.50, 1 - pcnt])
    iqr = qhigh - qlow
    return sr[np.abs((sr - median)) <= iqr]


def convert_neg_binom_params(mu, disp):
    """
    converts mean, std to n and p of scipy.negbinom rv

    See https://anton-granik.medium.com/fitting-and-visualizing-a-negative-binomial-distribution-in-python-3cc27fbc7ecf
    """

    p = 1 / (1 + mu * disp)
    n = mu * p / (1 - p)
    return n, p

