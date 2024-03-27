import jax

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import numpyro as npy
import numpyro.distributions as npd

from scipy import stats
from scipy.special import expit


def t_cell_simulation(n_clones=3,
                      mean_binder_range=None,
                      shape_binder_range=None,
                      n_cells_per_binder=None,
                      mean_non_binder=50,
                      shape_non_binder=5,
                      n_cells_per_non_binder=None,
                      binding_ratio=0.5,
                      rnd=42):
    """
    Generates test data for a single epitope assignment for n_clones with n_cells_per_clone following:

    1. Randomly assign  n_clones to binder or non-binder matching binding_ratio

    2. For each clone:
        A. for binder: 
            a1. draw mean randomly generally high "avidity" representing different binding strength
            a2. draw std randomly from low to moderate range representing clone-specific variance 
            a3. draw n_cell_per_clone uniformly from n_cell_per_binder
            a4. draw count data from negative binomial n_cells_per_clone times with generated parameters
        
        B. for non-binder: 
            b1. draw from an epitope-specific negative binomial, representing unspecific binding, with low mean and moderate std 
            b2. draw n_cell_per_clone uniformly from n_cell_per_non_binder
            b3. draw count data from negative binomial n_cells_per_clone times with generated parameters

            
    :n_clones: number of T cell clones
    :mean_binder_range: tuple with start and end range of binding avidity means
    :std_binder_range: tuple with start and end range of binding avidity standard deviation
    :n_cell_per_binder: range of sampled T cells per binding clone
    :mean_non_binder: mean avidity of non-binding T cell clones
    :std_non_binder: std avidity of non-binding T cell clones
    :binding_ratio: ratio of binding clones to non-binding clones
    :n_cell_per_non_binder: range of sampled T cells per non-binding clone (lower than n_cell_per_binder)
    :rnd: random seed

    return: two df (one epitope data and one neg controle) with n_clones*n_cells_per_clone rows and avidity,
            binary binding, and clonotype assignment as column
    """

    if n_cells_per_non_binder is None:
        n_cells_per_non_binder = [10, 100]
    if n_cells_per_binder is None:
        n_cells_per_binder = [500, 1000]
    if shape_binder_range is None:
        shape_binder_range = [1, 5]
    if mean_binder_range is None:
        mean_binder_range = [500, 510]

    np.random.seed(rnd)
    d = {"avidity": [], "binder": [], "clone": [], "size_factor": []}
    d_neg = {"avidity": [], "binder": [], "clone": [], "size_factor": []}
    binder_assignment = np.random.binomial(1, binding_ratio, size=n_clones)

    for i in range(n_clones):
        is_binder = binder_assignment[i]

        if is_binder:
            n_cell = np.random.randint(*n_cells_per_binder, size=1)[0]
            mean = np.random.uniform(*mean_binder_range, size=1)[0]
            shape = np.random.uniform(*shape_binder_range, size=1)[0]
            d["avidity"].extend(generate_nb_val(mean, shape, size=n_cell))
            d["binder"].extend([is_binder] * n_cell)
            d["clone"].extend([i] * n_cell)
        else:
            n_cell = np.random.randint(*n_cells_per_non_binder, size=1)[0]
            d["avidity"].extend(generate_nb_val(mean_non_binder, shape_non_binder, size=n_cell))
            d["binder"].extend([is_binder] * n_cell)
            d["clone"].extend([i] * n_cell)

        n_cell = np.random.randint(*n_cells_per_non_binder, size=1)[0]
        d_neg["avidity"].extend(generate_nb_val(mean_non_binder, shape_non_binder, size=n_cell))
        d_neg["binder"].extend([0] * n_cell)
        d_neg["clone"].extend([i] * n_cell)

    return pd.DataFrame.from_dict(d), pd.DataFrame.from_dict(d_neg)


def simulate_sc_pmhc_data(total_cells: int = 5000,
                          nof_clones: int = 150,
                          binding_ratio: float = 0.05,
                          neg_mean: float = 4.710658073425293,
                          neg_concentration: float = 0.393927070346017,
                          binding_fold_increase_range=None,
                          inv_concentration_range=None,
                          size_factor_param=None,
                          cells_per_binder_param=None,
                          cells_per_nonbinder_param=None,
                          clonotype_covariance: np.array = None,
                          generate_neg_x: bool = False,
                          rnd: int = 42
                          ) -> sc.AnnData:
    """
    Given negative control mean and concentration parameters (estimated from real data) generate binding data with
    predefined positive fold-change. All other parameters should be also realistic.

    Args:
        total_cells: number of total cell to generate
        nof_clones: number of clones measured in experiments.
        binding_ratio: ratio of binder vs non-binder
        neg_mean: the mean parameter of a Negative Binomial distribution fitted against negative control
                   pMHC-dextramer data.
        neg_concentration: the concentration parameter of a Negative Binomial distribution fitted against negative
                           control pMHC-dextramer data.
        binding_fold_increase_range: a list of positive fold-changes based on the neg_mean for `binding` data from which
                                   randomly will be selected.
        inv_concentration_range: a tuple of start and end ranges of inverse concentration parameters from which randomly
                                 samples will be generated for each clone.
        size_factor_param: a triple of lognormal params with which library sizes are drawn per cell.
        cells_per_binder_param: loc and scale parameter of exponential distribution fitted against clone size data
        cells_per_nonbinder_param: loc and scale parameter of exponential distribution fitted against clone size data (<=10)
        clonotype_covariance: covariance matrix based on sequence-distances of clonotype.
        generate_neg_x: boolean indicating whether negative control samples should be generated for each clone.
        rnd: random seed.

    Returns:
        An Anndata object containing all generated count data and clonal information, size_factors, and binder status
    """

    np.random.seed(rnd)

    if cells_per_nonbinder_param is None:
        cells_per_nonbinder_param = [1, 1.759259259259259]
    if cells_per_binder_param is None:
        cells_per_binder_param = [11.0, 179.78571428571428]
    if inv_concentration_range is None:
        inv_concentration_range = [0.001, 2.5]
    if binding_fold_increase_range is None:
        binding_fold_increase_range = [0.5, 1, 5, 10, 50, 100, 150, 200]
    if size_factor_param is None:
        size_factor_param = [0.8, 1.3]

    if clonotype_covariance is None:
        binder_assignment = np.random.binomial(1, binding_ratio, size=nof_clones)
    else:
        # inducing correlation structure TODO: might need to add noise here?
        p_clone = expit(np.random.multivariate_normal(mean=np.zeros(nof_clones), cov=clonotype_covariance))
        binder_assignment = np.random.binomial(1, p_clone)

    total_le = total_cells - nof_clones
    raw_cells_per_clone = np.array(stats.expon.rvs(*cells_per_binder_param)
                                   if binder_assignment[i] else stats.expon.rvs(*cells_per_nonbinder_param)
                                   for i in range(nof_clones))
    raw_cells_per_clone_norm = raw_cells_per_clone/raw_cells_per_clone.sum()
    cells_per_clone = np.random.multinomial(total_le, raw_cells_per_clone_norm) + np.ones(nof_clones)

    d = {"x": [], "x_neg": [], "binder": [], "clone": [],
         "size_factor": [], "fold_increase": [], "mean": [], "concentration": []}

    for i in range(nof_clones):
        is_binder = binder_assignment[i]
        n_cells = cells_per_clone[i]
        size_factor = stats.lognorm.rvs(size_factor_param, size=n_cells)

        if is_binder:
            fold_change = np.random.choice(*binding_fold_increase_range)
            mean = size_factor * (neg_mean + fold_change*neg_mean)
            concentration = 1.0/np.random.uniform(*inv_concentration_range)
            x = generate_nb_val(mean, concentration, size=n_cells)
        else:
            fold_change = 0
            mean = size_factor * neg_mean
            a = (0.001 - neg_concentration) / (neg_concentration/3)
            concentration = stats.truncnorm.rvs(a, np.inf, loc=neg_concentration, scale=neg_concentration/3)
            x = generate_nb_val(mean, concentration, size=n_cells)

        d["x"].extend(x)
        d["binder"].extend([is_binder] * n_cells)
        d["clone"].extend([i] * n_cells)
        d["size_factor"].extend(size_factor)
        d["fold_increase"].extend([fold_change] * n_cells)
        d["mean"].extend(mean)
        d["concentration"].extend([concentration] * n_cells)
        d["x_neg"].extend([np.NaN]*n_cells)

        if generate_neg_x:
            n_cells = np.random.randint(*cells_per_nonbinder_param)
            size_factor = np.random.uniform(*size_factor_param)
            fold_change = 0
            mean = size_factor * neg_mean
            concentration = neg_concentration
            x_neg = generate_nb_val(mean, concentration, size=n_cells)

            d["x"].extend([np.NaN] * n_cells)
            d["binder"].extend([0] * n_cells)
            d["clone"].extend([i] * n_cells)
            d["size_factor"].extend([size_factor] * n_cells)
            d["fold_increase"].extend([fold_change] * n_cells)
            d["mean"].extend([mean] * n_cells)
            d["concentration"].extend([concentration] * n_cells)
            d["x_neg"].extend([x_neg] * n_cells)

    adata = ad.AnnData(np.array([d["x"], d["x_neg"]], dtype="float64").T)
    adata.var_names = ["epitope1", "neg_control"]
    adata.obs["size_factor"] = d["size_factor"]
    adata.obs["binder"] = d["binder"]
    adata.obs["clone"] = d["clone"]
    adata.obs["fold_increase"] = d["fold_change"]
    adata.obs["mean"] = d["mean"]
    adata.obs["concentration"] = d["concentration"]

    return adata


def generate_nb_val(mu, alpha, size=1, rng_key=42):
    """Generate negative binomial samples

    Args:
        mu: the mean parameter (must be positive)
        alpha: the inverse overdispersion parameter (must be positive)
        size: the number of iid draws
        rng_key: an integer to initialize the random key generator.
    """
    return npd.NegativeBinomial2(mu, alpha).sample(jax.random.PRNGKey(rng_key), sample_shape=(size,))


def convert_neg_bino_params(mu, std):
    """
    converts mean, std to n and p of scipy.negbinom rv

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """

    p = mu / std ** 2
    n = mu * p / (1.0 - p)
    return n, p
