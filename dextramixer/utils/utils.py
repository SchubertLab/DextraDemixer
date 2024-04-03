import jax
import jax.numpy as jnp


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


def sample_orthogonal_mtx(n: int) -> jax.Array:
    """
    samples an orthogonal matrix of size nxn

    Args:
        n: dimension size of matrix
    Returns:
        A nxn orthonormal matrix
    """
    A = jax.random.normal(shape=(n, n))
    Q, _ = jnp.linalg.qr(A)
    return Q


def sample_cov_from_eigs(eigs: jax.Array) -> jax.Array:
    """
    samples a covariance matrix sampling an orthogonal matrix and multiplying it with eigenvalues
    Args:
        eigs: a list of eigenvalues of size n
    Returns:
        a covariance matrix of size nxn
    """
    E = jnp.diag(eigs)
    P = sample_orthogonal_mtx(eigs.shape[0])
    return jnp.matmul(jnp.matmul(P, E), P.T)


def remove_outliers(sr, iq_range=0.8):
    #  https://stackoverflow.com/a/39424972
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1 - pcnt])
    iqr = qhigh - qlow
    return sr[(sr - median).abs() <= iqr].to_numpy()
