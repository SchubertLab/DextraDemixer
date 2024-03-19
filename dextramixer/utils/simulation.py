import numpy as np
import pandas as pd
from scipy import stats


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
    d = {"avidity": [], "binder": [], "clone": []}
    d_neg = {"avidity": [], "binder": [], "clone": []}
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


def generate_nb_val(mu, alpha, size):
    """Generate negative binomial distributed samples by
    drawing a sample from a gamma distribution with mean `mu` and
    shape parameter `alpha`, then drawing from a Poisson
    distribution whose rate parameter is given by the sampled
    gamma variable.
    """
    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    if len(g) <= 1:
        return [stats.poisson.rvs(g)]
    return stats.poisson.rvs(g)


def convert_neg_bino_params(mu, std):
    """
    converts mean, std to n and p of scipy.negbinom rv

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """

    p = mu / std ** 2
    n = mu * p / (1.0 - p)
    return n, p
