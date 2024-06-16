from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy import ndarray, dtype, bool_
from scipy.stats import ortho_group, random_correlation


def gower_centering(distance_matrix):
    """
    Applies Gower's 1966 centering method to the distance matrix to obtain a covariance matrix.

    Parameters:
        distance_matrix (jax.numpy.ndarray): Symmetric distance matrix.

    Returns:
        jax.numpy.ndarray: Covariance matrix.
    """
    n = distance_matrix.shape[0]

    # Compute the squared distance matrix
    squared_distances = jnp.square(distance_matrix)

    # Compute the row means, column means, and total mean of the squared distance matrix
    row_means = jnp.mean(squared_distances, axis=1, keepdims=True)
    col_means = jnp.mean(squared_distances, axis=0, keepdims=True)
    total_mean = jnp.mean(squared_distances)

    # Apply the Gower centering formula
    return -0.5 * (squared_distances - row_means - col_means + total_mean)


def nearest_psd(matrix, thresh=0.0, use_abs=False):
    """
    Adjusts a matrix to ensure it is positive semi-definite using JAX.

    Parameters:
        matrix (jax.numpy.ndarray): Input matrix.
        use_abs (bool): specify if eigenvalues are adjusted by taking the absolut values or setting negative values to 0
                        (default False)
    Returns:
        jax.numpy.ndarray: Adjusted positive semi-definite matrix.
    """
    matrix = (matrix + matrix.T) / 2

    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    # Set negative eigenvalues to zero or abs
    if use_abs:
        eigenvalues = jnp.abs(eigenvalues)
    else:
        eigenvalues = jnp.where(eigenvalues < thresh, 1e-10, eigenvalues)
    return eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T


def dist_to_cov_psd(d, use_abs=False):
    """
    Converts a symmetric distance matrix into a symmetric positive semi-definite covariance matrix using Gower's
    centering method.

    Args:
        d (jax.numpy.ndarray): Symmetric distance matrix.
        use_abs (bool): specify if eigenvalues are adjusted by taking the absolut values or setting negative values to 0
                        (default False)
    Returns:
        jax.numpy.ndarray: Symmetric positive semi-definite covariance matrix.
    """
    return nearest_psd(gower_centering(d), use_abs)


def normalize_distance_matrix(D):
    min_val = jnp.min(D)
    max_val = jnp.max(D)
    return (D - min_val) / (max_val - min_val)


def dist_to_sim(d, normalize=True, sigma=None, epsilon=None):
    """"
    Converts a symmetric distance matrix into a symmetric positive semi-definite similarity matrix using an RBF Kernel:

    Kij = exp(- Dij^2/(2*sigma^2))

    Args:
        d (jax.numpy.ndarray): Symmetric distance matrix.
        normalize (bool: indicating whether  Min-Max normalize should be applied
        sigma (float): the hyperparameter of the RBF Kernel, if None then the median of the non-zero elements will be used
        epsilon(float): a small float 1e-6 that is added to the diagonal of the similarity matrix to stabilize it
    Returns:
        jax.numpy.ndarray: Symmetric positive semi-definite covariance matrix.
    """
    distance_matrix = jnp.array(d).astype("float64")
    if normalize:
        distance_matrix = normalize_distance_matrix(distance_matrix)

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if not jnp.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric")

    if sigma is None:
        sigma = jnp.median(distance_matrix[jnp.nonzero(distance_matrix)])

    K = jnp.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
    if epsilon is not None:
        K += epsilon * jnp.eye(K.shape[0])
    return nearest_psd(K)


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
    eigs = jnp.where(eigs < 0, 0, eigs)
    S = jnp.diag(eigs)
    Q = sample_orthogonal_mtx(eigs.shape[0], rng_key=rng_key)
    return Q.T @ S @ Q


def generate_sim_from_ltridist(ltrdist, normalize=False, sigma=None):
    """
    generates a symmetric similarity matrix given a lower triangular matrix of distances

    """
    N = jnp.int32((jnp.sqrt(8 * jnp.size(ltrdist) + 1) + 1) / 2)

    # Reshape flat array into lower triangular matrix

    tril_indices = jnp.tril_indices(N, -1)

    # Create the full distance matrix by mirroring the lower triangle
    distance_matrix = jnp.zeros((N, N))
    distance_matrix = distance_matrix.at[tril_indices].set(ltrdist)
    distance_matrix = distance_matrix.at[(tril_indices[1], tril_indices[0])].set(ltrdist)

    sim = dist_to_sim(distance_matrix, normalize=normalize, sigma=sigma)

    return sim


def sample_corr_from_eigen(eigs: jax.Array, rng_key: int = 42) -> jax.Array:
    return random_correlation.rvs(eigs, random_state=rng_key)


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


def convert_to_variance(mu, disp):
    """
    converts mean and dispersion of negative binomial to variance
    """
    return mu + disp*mu**2


def convert_to_invdispersion(mu, var):
    """
    converts mu and variance to inverse dispersion param of negative binomial
    """
    return 1/((var - mu)/mu**2)
