import itertools
import os
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scirpy as ir

from jax import pure_callback
from numpy import ndarray, dtype, bool_
from scipy.stats import ortho_group, random_correlation
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, matthews_corrcoef)

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
        eigenvalues = jnp.where(eigenvalues < thresh, 1e-6, eigenvalues)
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


def dist_to_sim(d, nearest_psed=False, normalize=True, sigma=None, epsilon=None):
    """"
    Converts a symmetric distance matrix into a symmetric positive semi-definite similarity matrix using an RBF Kernel:

    Kij = exp(- Dij^2/(2*sigma^2))

    Args:
        d (jax.numpy.ndarray): Symmetric distance matrix.
        nearest_psed (bool): indicating whether the nearest PSD matrix should be constructed
        normalize (bool): indicating whether  Min-Max normalize should be applied
        sigma (float): the hyperparameter of the RBF Kernel, if None then the median of the non-zero elements will be used
        epsilon(float): a small float 1e-6 that is added to the diagonal of the similarity matrix to stabilize it
    Returns:
        jax.numpy.ndarray: Symmetric positive semi-definite covariance matrix.
    """
    distance_matrix = jnp.array(d)
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
    if nearest_psed:
        K = nearest_psd(K)
    return K


def calculate_clonotype_kernel(mdat,
                               distance="tcrdist",
                               ir_key="airr",
                               key_added="dextrademixer",
                               normalize=False,
                               sigma=None,
                               nearest_psed=False,
                               epsilon=1e-8):
    """
        calculates TCR dist based on specified metric for unique clones (based on aa sequence identity),
        calculates kernel based on that, and stores the kernel under specified `airr`.uns.
    """
    ir.pp.ir_dist(mdat, metric="identity", sequence="aa", cutoff=int(1e8))
    ir.pp.ir_dist(mdat, metric=distance, sequence="aa", cutoff=int(1e8))

    _, _, d_ident = ir.tl.define_clonotype_clusters(mdat, sequence="aa", metric="identity",
                                              receptor_arms="all", dual_ir="any", inplace=False)
    _, _, d_dist = ir.tl.define_clonotype_clusters(mdat, sequence="aa", metric=distance,
                                             receptor_arms="all", dual_ir="any", inplace=False)

    # check whether rows are ordered equally if not calculate permutation,
    cc_ident = d_ident["cell_indices"]
    cc_dist = d_dist["cell_indices"]

    permutation = np.arange(d_dist["distances"].shape[0])
    loners = []

    for k_dist, val_dist in cc_dist.items():
        cc_ident_val = cc_ident[k_dist]
        if len(set(cc_ident_val) - set(val_dist)) == 0:
            continue
        else:
            for k_id, val_id in cc_ident.items():
                if len(set(val_id) - set(val_dist)) == 0:
                    permutation[k_dist] = k_id
                    break

            loners.append(k_dist)

    if loners:
        raise RuntimeError("Discrepancies between clonal identity and distance identity detected")

    # define clonotype id - cc_identity is reference
    idx, values = zip(
        *itertools.chain.from_iterable(
            zip(cell_ids, itertools.repeat(str(clonotype_cluster)))
            for clonotype_cluster, cell_ids in cc_ident.items()
        ),
        strict=False,
    )
    clonotype_cluster_series = pd.Series(values, index=idx).reindex(mdat.mod[ir_key].obs_names)
    clonotype_cluster_size_series = clonotype_cluster_series.groupby(clonotype_cluster_series).transform("count")

    # extract distance
    dist = d_dist["distances"].todense() - 1  # see scirpy sparse_matrix def
    dist = dist[permutation]
    K = dist_to_sim(dist, nearest_psed=nearest_psed, normalize=normalize, sigma=sigma, epsilon=epsilon)

    mdat.mod[ir_key].uns[f"{key_added}_distances"] = dist
    mdat.mod[ir_key].uns[f"{key_added}_kernel"] = K
    mdat.mod[ir_key].obs[f"{key_added}_clone_id"] = clonotype_cluster_series
    mdat.mod[ir_key].obs[f"{key_added}_clone_id_size"] = clonotype_cluster_size_series


def sim_to_dist(s: jax.Array) -> jax.Array:
    """
    converts a quadratic similarity matrix into a distance matrix
    """
    if s.shape[0] != s.shape[1] or jnp.any(s != s.T):
        raise ValueError(f"Similarity matrix must be square and symmetric.")

    EPS = jnp.finfo("float32").eps
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
    rng = np.random.RandomState(seed=rng_key)
    return ortho_group.rvs(dim=n, random_state=rng)


def sample_cov_from_eigs(eigs: jax.Array, rng_key: int = 42) -> ndarray[Any, dtype[bool_]]:
    """
    samples a covariance matrix sampling an orthogonal matrix and multiplying it with eigenvalues
    Args:
        eigs: a list of eigenvalues of size n
        rng_key: a random seed
    Returns:
        a covariance matrix of size nxn
    """
    eigs = jnp.where(eigs < 0, 1e-8, eigs)
    S = jnp.diag(eigs)
    Q = sample_orthogonal_mtx(eigs.shape[0], rng_key=rng_key)
    return Q.T @ S @ Q


def generate_sim_from_ltridist(ltrdist, normalize=False, sigma=None, epsilon=0.0):
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
    sim = dist_to_sim(distance_matrix, normalize=normalize, sigma=sigma, epsilon=epsilon)

    return sim


def sample_corr_from_eigen(eigs: jax.Array, rng_key: int = 42) -> jax.Array:
    rng = np.random.RandomState(seed=rng_key)
    return random_correlation.rvs(eigs, random_state=rng)


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
    return mu + disp * mu ** 2


def convert_to_invdispersion(mu, var):
    """
    converts mu and variance to inverse dispersion param of negative binomial
    """
    return 1 / ((var - mu) / mu ** 2)


def hook_optax(optimizer):
    """
    Helper function to collect gradient norms during training
    """
    gradient_norms = defaultdict(list)

    def append_grad(grad):
        for name, g in grad.items():
            gradient_norms[name].append(float(jnp.linalg.norm(g)))
        return grad

    def update_fn(grads, state, params=None):
        grads = pure_callback(append_grad, grads, grads)
        return optimizer.update(grads, state, params=params)

    return optax.GradientTransformation(optimizer.init, update_fn), gradient_norms


def convert_str_to_bool_and_none(args):
    def str_to_bool(s):
        if not isinstance(s, str):
            return s
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s
    for key, value in vars(args).items():
        setattr(args, key, str_to_bool(value))

    return args


def float_or_none(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid float or 'None'")
    

def get_slurm_cpu_count():
    # Check for SLURM-provided variables
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            value = os.environ[var]
            # SLURM_JOB_CPUS_PER_NODE can be something like "4(x2)" meaning 2 nodes with 4 CPUs each
            if "(" in value:
                value = value.split("(")[0]
            try:
                return int(value)
            except ValueError:
                pass
    # Fallback
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def guess_worker_mem_limit_mb(nworkers: int):
    # If SLURM ressources are present
    if "SLURM_MEM_PER_NODE" in os.environ:
        return int(int(os.environ["SLURM_MEM_PER_NODE"]) * 0.95 // nworkers)
    if "SLURM_MEM_PER_CPU" in os.environ:
        return int(int(os.environ["SLURM_MEM_PER_CPU"]) * 0.95)
    return None  # no good signal; skip limiting


def init_worker(worker_mem_limit_mb=None):
    if worker_mem_limit_mb is None:
        return
    try:
        import resource
        limit_bytes = int(worker_mem_limit_mb) * 1024 * 1024
        # Address space cap → allocations above this raise MemoryError
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        # If we can't set it, just proceed; kernel OOM may still occur.
        pass


def calculate_metrics(y_true: np.ndarray, p_pred: np.ndarray, assignment: np.ndarray) -> dict:
    results_dict = {'auroc': roc_auc_score(y_true, p_pred), 'aps': average_precision_score(y_true, p_pred),
                    'f1': f1_score(y_true, assignment), 'precision': precision_score(y_true, assignment),
                    'recall': recall_score(y_true, assignment), 'accuracy': accuracy_score(y_true, assignment),
                    'mcc': matthews_corrcoef(y_true, assignment)}

    return results_dict
