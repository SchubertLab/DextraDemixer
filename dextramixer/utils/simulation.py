import collections.abc
import numbers
import warnings
from typing import Union, Tuple, Optional

import jax

import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import mudata as md
import numpyro as npy
import numpyro.distributions as npd
import statsmodels.formula.api as smf

from scipy import stats
from scipy.special import expit

import matplotlib.pyplot as plt

from dextramixer.utils.utils import sample_cov_from_eigs, dist_to_cov, remove_outliers, convert_neg_binom_params


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


def t_cell_simulation(n_clones=3,
                      mean_binder_range=None,
                      shape_binder_range=None,
                      n_cells_per_binder=None,
                      mean_non_binder=50,
                      shape_non_binder=5,
                      n_cells_per_non_binder=None,
                      binding_ratio=0.5,
                      rng_key=42):
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
            b1. draw from an epitope-specific negative binomial, representing unspecific binding, with low mean and
                moderate std
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
    :rng_key: random seed

    return: two df (one epitope data and one neg control) with n_clones*n_cells_per_clone rows and avidity,
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

    np.random.seed(rng_key)
    d = {"avidity": [], "binder": [], "clone": [], "size_factor": []}
    d_neg = {"avidity": [], "binder": [], "clone": [], "size_factor": []}
    binder_assignment = np.random.binomial(1, binding_ratio, size=n_clones)

    for i in range(n_clones):
        is_binder = binder_assignment[i]

        if is_binder:
            n_cell = np.random.randint(*n_cells_per_binder, size=1)[0]
            mean = np.random.uniform(*mean_binder_range, size=1)[0]
            shape = np.random.uniform(*shape_binder_range, size=1)[0]
            d["avidity"].extend(DextramerSimulator.generate_nb_val(mean, shape, size=n_cell).tolist())

        else:
            n_cell = np.random.randint(*n_cells_per_non_binder, size=1)[0]
            d["avidity"].extend(
                DextramerSimulator.generate_nb_val(mean_non_binder, shape_non_binder, size=n_cell).tolist())

        d["binder"].extend([is_binder] * n_cell)
        d["clone"].extend([i] * n_cell)
        d["size_factor"].extend(np.ones(n_cell))

        d_neg["avidity"].extend(DextramerSimulator.generate_nb_val(mean_non_binder,
                                                                   shape_non_binder, size=n_cell).tolist())
        d_neg["binder"].extend([0] * n_cell)
        d_neg["clone"].extend([i] * n_cell)
        d_neg["size_factor"].extend(np.ones(n_cell))

    adata = ad.AnnData(np.array([d["avidity"], d_neg["avidity"]], dtype="float64").T)
    adata.var_names = ["pmhc1", "neg_control"]
    adata.var["feature_types"] = ["Antigen Capture", "Antigen Capture"]
    adata.obs["size_factor"] = d["size_factor"]

    adata_tcr = ad.AnnData()
    adata_tcr.obs["is_binder"] = d["binder"]
    adata_tcr.obs["clone_id"] = d["clone"]
    adata_tcr.uns["ir_cov"] = np.eye(len(np.unique(d["clone"])))

    return md.MuData({"gex": adata, "airr": adata_tcr})


class DextramerSimulator:
    """
    Simulates dextramer single-cell data based on inferred parameters from real experiments
    """

    def __init__(self):
        self.params = None

    @staticmethod
    def default_params():
        default_params = {
            "neg_mean": 4.710658073425293,
            "neg_concentration": 0.393927070346017,
            "size_factor_param": (0.42792255, -262.7462615966805, 1361.486),
            "cells_per_binder_param": [0.005516163158815324, 3559.0, 10.0],
            "cells_per_nonbinder_param": [0.450069425326169, 1874.0, 1.0],
            "concentration_param": (0.6017693693535755, 0.09382864854673992, 3.0634293748626753),  # [0.001, 2.5],
            "clonotype_eigs_param": (-155.18652967415233, 532.7534635778651),
        }
        return default_params

    def estimate_simulation_params(self,
                                   mdata: md.MuData,
                                   neg_ctrl_key: str,
                                   gex_key: str = "gex",
                                   ir_key: str = "airr",
                                   ir_dist_key: str = "dist",
                                   boltzmann_boundary: Tuple[int, int] = (0, 10000),
                                   filter_extreme_values: Union[bool, list[bool]] = False,
                                   iq_range: Union[float, list[float]] = 0.8,
                                   plot_qc: bool = False,
                                   rng_key: int = 42) -> Optional[plt.Axes]:
        """
        Estimates necessary parameters from real world pMHC data. Requires a negative control pMHC dextramer
        and known clonotype ids and clonotype distances based on some distance measure.

        Only QC filtering should have been performed but now normalization yet

        Args:
            mdata: A Mudata containing only dextramer counts and clonotype information
            neg_ctrl_key: a string specifying the negative control column
            gex_key: the MuData transcriptome module key
            ir_key: the MuData AIRR module key
            ir_dist_key: the key in AIRR module's '.uns' that contains a full, symmetric and square distance matrix
                         for all clonotype cluster
            boltzmann_boundary: a tuple of floats representing the boundary conditions of a discrete Boltzmann
                                distribution
            filter_extreme_values: boolean or list of booleans indicating whether extreme values should be filtered
                                   before fitting the theoretical distributions. If a list is provided, at least five
                                   booleans, one per fitted category of distributions, must be provided.
            iq_range: inter-quantile range or list of iqr range used to determine extreme values
                      (Only used if `filter_extreme_values` = True). If a list is provided, at least five
                                   iqr, one per fitted category of distributions, must be provided.
            plot_qc: bool determining whether to generate QC-plots for each theoretical dist
            rng_key: random seed.
        Returns:
            (Optional) Matplotlib.Axis array if `plot_qc` = True
        """
        np.random.seed(rng_key)
        i = 0

        def __remove_extreme_values(x, is_filter, iqr):
            nonlocal i
            i += 1
            return remove_outliers(x, iqr) if is_filter else x

        if not isinstance(mdata, md.MuData):
            raise ValueError("`mdat`is not a MuData object. Please read the scirpy tutorial to combine GEX and AIRR "
                             "data.")

        if isinstance(filter_extreme_values, bool):
            filter_extreme_values = [filter_extreme_values]*5
        if isinstance(filter_extreme_values, collections.abc.Collection) and len(filter_extreme_values) < 5:
            raise ValueError("`filter_extreme_values` must have a length of at least five.")

        if isinstance(iq_range, numbers.Number) and not isinstance(iq_range, bool):
            iq_range = [iq_range]*5

        if isinstance(iq_range, collections.abc.Collection) and len(iq_range) < 5:
            raise ValueError("`iq_range` must have a length of at least five.")

        param = {}

        # normalize gex data
        X_norm, size_factor = sc.pp.normalize_total(mdata.mod[gex_key], inplace=False).values()
        neg_idx = mdata.mod[gex_key].var["gene_ids"].to_list().index(neg_ctrl_key)

        #####################
        # Estimate parameters
        #####################
        neg_x = __remove_extreme_values(X_norm[:, neg_idx].toarray()[:, 0], filter_extreme_values[i], iq_range[i])
        size_factor = __remove_extreme_values(size_factor, filter_extreme_values[i], iq_range[i])

        # estimation of mean and inverse dispersion parameter from nb model
        nbfit = smf.negativebinomial("nbdata ~ 1",
                                     data=pd.DataFrame({"nbdata": neg_x})).fit(disp=False)

        param["neg_mean"] = np.exp(nbfit.params.iloc[0])
        param["neg_concentration"] = 1 / nbfit.params.iloc[1]

        # fit size factor distribution
        param["size_factor_param"] = stats.lognorm.fit(size_factor)

        # fit clonotype size distribution
        clone_size = __remove_extreme_values(mdata.mod[ir_key].obs.groupby("clone_id", dropna=False).size(),
                                             filter_extreme_values[i], iq_range[i])
        q80_clone_size = np.quantile(clone_size, 0.8)
        rv = stats.boltzmann
        bounds_low = [boltzmann_boundary, boltzmann_boundary, (1, 1)]
        bounds_high = [boltzmann_boundary, boltzmann_boundary, (q80_clone_size, q80_clone_size)]
        res_low = stats.fit(rv, clone_size[clone_size <= q80_clone_size], bounds_low)
        res_high = stats.fit(rv, clone_size[clone_size > q80_clone_size], bounds_high)

        if not res_low.success:
            warnings.warn("Estimation of boltzmann parameters on the lower 80-quantile of clone sizes failed. Please "
                          "adjust boundary conditions of the parameters")
        if not res_high.success:
            warnings.warn("Estimation of boltzmann parameters on the upper 80-quantile of clone sizes failed. Please "
                          "adjust boundary conditions of the parameters")

        param["cells_per_nonbinder_param"] = list(res_low.params)
        param["cells_per_binder_param"] = list(res_high.params)

        # fit inv dispersion distribution
        invdisp = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for c, g in mdata.mod[ir_key].obs.groupby("clone_id", dropna=False):
                if g.shape[0] < 15:
                    continue
                m = mdata.mod["gex"][g.index]
                for j in m.var.gene_ids:
                    d = m[:, j].to_df()
                    d = d.rename({j: j.replace("-", "_")}, axis=1)
                    nbfit = smf.negativebinomial(f"{d.columns[0]} ~ 1", data=d).fit(disp=False)
                    if not nbfit.converged:
                        continue
                    invdisp.append(1 / nbfit.params.iloc[1])  # concentration parameter

        invdisp = __remove_extreme_values(np.array(invdisp), filter_extreme_values[i], iq_range[i])
        param["concentration_param"] = stats.gamma.fit(invdisp)

        # fit prior for covariance matrix
        dist = mdata.mod[ir_key].uns[ir_dist_key]

        cov = dist_to_cov(dist)
        eigs = __remove_extreme_values(np.real(np.linalg.eigvals(cov)), filter_extreme_values[i], iq_range[i])
        param["clonotype_eigs_param"] = stats.semicircular.fit(eigs)

        self.params = param

        # QQ plot
        if plot_qc:
            return self.__qc_plot(neg_x, size_factor, clone_size, q80_clone_size, invdisp, dist, cov, eigs, rng_key)

    def __qc_plot(self, neg_x, size_factor, clone_size, q80_clone_size, invdisp, dist, cov, eigs, rng_key):
        """
        Plots QQ plots of fitted theoretical distribution against empirical distribution
        """

        if self.params is not None:
            params = {**DextramerSimulator.default_params(), **self.params}
        else:
            params = DextramerSimulator.default_params()

        fig_params = {'legend.fontsize': 'x-small',
                      'figure.figsize': (8.27, 11.69),
                      'figure.dpi': 100,
                      'axes.labelsize': 'x-small',
                      'axes.titlesize': 'x-small',
                      'xtick.labelsize': 'x-small',
                      'ytick.labelsize': 'x-small'
                      }
        plt.rcParams.update(fig_params)
        blue = sns.color_palette("tab10", 10)[0]
        sample_size = 5000

        fig, axs = plt.subplots(7, 3, layout='tight', gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 2]})

        sns.histplot(neg_x, log_scale=True, legend=False, ax=axs[0, 0])
        negbinom_params = convert_neg_binom_params(params["neg_mean"], 1 / params["neg_concentration"])
        axs[0, 0].title.set_text("Empirical negative control distribution")
        stats.probplot(neg_x, dist=stats.nbinom, sparams=negbinom_params, plot=axs[0, 1], rvalue=True)
        axs[0, 1].get_children()[2].set_fontsize("x-small")
        axs[0, 1].title.set_text("Negative Binomial fitted negative control")
        axs[0, 1].get_lines()[0].set_color(blue)
        sns.histplot(stats.nbinom.rvs(*negbinom_params, size=sample_size), log_scale=True, legend=False, ax=axs[0, 2])
        axs[0, 2].title.set_text("Fitted negative control distribution")

        sns.histplot(x=size_factor, log_scale=False, legend=False, ax=axs[1, 0])
        axs[1, 0].title.set_text("Empirical size factor distribution")
        stats.probplot(size_factor, dist=stats.lognorm,
                       sparams=params["size_factor_param"], plot=axs[1, 1], rvalue=True)
        axs[1, 1].get_children()[2].set_fontsize("x-small")
        axs[1, 1].get_lines()[0].set_color(blue)
        axs[1, 1].title.set_text("Lognormal fitted size factor")
        sns.histplot(stats.lognorm.rvs(*params["size_factor_param"], size=sample_size),
                     log_scale=False, legend=False, ax=axs[1, 2])
        axs[1, 2].title.set_text("Fitted size factor distribution")

        sns.histplot(clone_size[clone_size > q80_clone_size], log_scale=True, legend=False, ax=axs[2, 0])
        axs[2, 0].title.set_text("Empirical clone size distribution $>$ q80")
        stats.probplot(clone_size[clone_size > q80_clone_size], dist=stats.boltzmann,
                       sparams=params["cells_per_binder_param"], plot=axs[2, 1], rvalue=True)
        axs[2, 1].get_children()[2].set_fontsize("x-small")
        axs[2, 1].title.set_text("Discrete Boltzmann fitted clone size")
        axs[2, 1].get_lines()[0].set_color(blue)
        sns.histplot(stats.boltzmann.rvs(*params["cells_per_binder_param"], size=sample_size),
                     log_scale=True, legend=False, ax=axs[2, 2])
        axs[2, 2].title.set_text("Fitted clone size distribution")

        sns.histplot(clone_size[clone_size <= q80_clone_size], log_scale=True, legend=False, ax=axs[3, 0])
        axs[3, 0].title.set_text("Empirical clone size distribution $\leq$ q80")
        stats.probplot(clone_size[clone_size <= q80_clone_size], dist=stats.boltzmann,
                       sparams=params["cells_per_nonbinder_param"], plot=axs[3, 1], rvalue=True)
        axs[3, 1].get_children()[2].set_fontsize("x-small")
        axs[3, 1].title.set_text("Discrete Boltzmann fitted clone size")
        axs[3, 1].get_lines()[0].set_color(blue)
        sns.histplot(stats.boltzmann.rvs(*params["cells_per_nonbinder_param"], size=sample_size),
                     log_scale=True, legend=False, ax=axs[3, 2])
        axs[1, 2].title.set_text("Fitted clone size distribution")

        sns.histplot(x=invdisp, log_scale=False, legend=False, ax=axs[4, 0])
        axs[4, 0].title.set_text("Empirical inverse dispersion \n distribution of clonotypes")
        stats.probplot(size_factor, dist=stats.gamma,
                       sparams=params["concentration_param"], plot=axs[4, 1], rvalue=True)
        axs[4, 1].title.set_text("Gamma fitted inverse dispersion \n of clonotypes")
        axs[4, 1].get_children()[2].set_fontsize("x-small")
        axs[4, 1].get_lines()[0].set_color(blue)
        axs[4, 2].title.set_text("Fitted inverse dispersion \n distribution of clonotypes ")
        sns.histplot(stats.gamma.rvs(*params["concentration_param"], size=sample_size),
                     log_scale=False, legend=False, ax=axs[4, 2])

        axs[5, 0].title.set_text("Empirical covariance \n eigenvalue distribution")
        sns.histplot(eigs, log_scale=True, ax=axs[5, 0])
        stats.probplot(eigs, dist="semicircular", sparams=params["clonotype_eigs_param"], plot=axs[5, 1], rvalue=True)
        axs[5, 1].title.set_text("Semicircle fitted eigenvalues")
        axs[5, 1].get_children()[2].set_fontsize("x-small")
        axs[5, 1].get_lines()[0].set_color(blue)
        axs[5, 2].title.set_text("Fitted eigenvalue distribution")
        sns.histplot(stats.semicircular.rvs(*params["clonotype_eigs_param"], size=sample_size),
                     log_scale=True, legend=False, ax=axs[5, 2])

        axs[6, 0].title.set_text("Distance matrix \n between clonotypes")
        sns.heatmap(dist, square=True, ax=axs[6, 0], cbar_kws={"shrink": 0.5})
        axs[6, 1].title.set_text("Covariance matrix \n between clonotypes")
        sns.heatmap(cov, square=True, ax=axs[6, 1], cbar_kws={"shrink": 0.5})
        eigs = stats.semicircular.rvs(*params["clonotype_eigs_param"], size=len(clone_size))
        cov_est = sample_cov_from_eigs(eigs, rng_key=rng_key)
        axs[6, 2].title.set_text("Eigenvalue simulated \n covariance matrix")
        sns.heatmap(cov_est, square=True, ax=axs[6, 2], cbar_kws={"shrink": 0.5})

        return axs

    def simulate_pmhc_data(self,
                           total_cells: int = 5000,
                           nof_clones: int = 150,
                           binding_ratio: float = 0.05,
                           binding_fold_increase_range: list[float] = None,
                           use_clonotype_cov: bool = False,
                           simulate_neg_control: bool = False,
                           rng_key: int = 42
                           ) -> md.MuData:
        """
        Given negative control mean and concentration parameters (estimated from real data) generate binding data for
        one pMHC with predefined positive fold-change.

        Args:
            total_cells: number of total cell to generate
            nof_clones: number of clones measured in experiments.
            binding_ratio: ratio of binder vs non-binder
            binding_fold_increase_range: list of fold increase for pMHC binding cells
            use_clonotype_cov: whether to use clonotype covariance to assign binding or randomly (default: False)
            simulate_neg_control: whether to simulate a negative control pMHC for each cell (default: False)
            rng_key: random seed.

        Returns:
            An Anndata object containing all generated count data and clonal information, size_factors, and binder status
        """

        np.random.seed(rng_key)

        if self.params is not None:
            params = {**DextramerSimulator.default_params(), **self.params}
        else:
            params = DextramerSimulator.default_params()

        # params
        neg_mean = params["neg_mean"]
        neg_concentration = params["neg_concentration"]
        clonotype_eigs_param = params["clonotype_eigs_param"]
        cells_per_binder_param = params["cells_per_binder_param"]
        cells_per_nonbinder_param = params["cells_per_nonbinder_param"]
        size_factor_param = params["size_factor_param"]
        concentration_param = params["concentration_param"]

        if binding_fold_increase_range is None:
            binding_fold_increase_range = [200]  # 0.5, 1, 5, 10, 50, 100, 150,

        if use_clonotype_cov:
            # sample covariance matrix
            eigs = stats.semicircular.rvs(*clonotype_eigs_param, size=nof_clones)
            cov = sample_cov_from_eigs(eigs, rng_key=rng_key)

            p_clone = expit(np.random.multivariate_normal(mean=np.zeros(nof_clones), cov=cov))
            binder_assignment = np.random.binomial(1, p_clone)
        else:
            binder_assignment = np.random.binomial(1, binding_ratio, size=nof_clones)

        # generate cell per clonotype following a discrete exponentially decreasing distribution normalized to
        # specified total cell count
        total_le = total_cells - nof_clones
        raw_cells_per_clone = np.array([stats.boltzmann.rvs(*cells_per_binder_param)
                                        if binder_assignment[i] else stats.boltzmann.rvs(*cells_per_nonbinder_param)
                                        for i in range(nof_clones)])
        cells_per_clone_p = stats.dirichlet.rvs(raw_cells_per_clone)[0]
        cells_per_clone = (np.random.multinomial(total_le, cells_per_clone_p) + np.ones(nof_clones)).astype("int32")

        d = {"x": [], "x_neg": [], "binder": [], "clone": [],
             "size_factor": [], "fold_increase": [], "mean": [], "concentration": []}

        for i in range(nof_clones):
            is_binder = binder_assignment[i]
            n_cells = cells_per_clone[i]
            size_factor = stats.lognorm.rvs(*size_factor_param, size=n_cells)

            if is_binder:
                fold_change = np.random.choice(binding_fold_increase_range)
                mean = size_factor * (neg_mean + fold_change * neg_mean)
                concentration = stats.gamma.rvs(*concentration_param)
            else:
                fold_change = 0
                mean = size_factor * neg_mean
                a = (0.001 - neg_concentration) / (neg_concentration / 3)
                concentration = stats.truncnorm.rvs(a, np.inf, loc=neg_concentration, scale=neg_concentration / 3)

            x = self.generate_nb_val(mean, concentration)

            if simulate_neg_control:
                mean = size_factor * neg_mean
                x_neg = self.generate_nb_val(mean, neg_concentration)
                d["x_neg"].extend(x_neg[0].tolist())

            d["x"].extend(x[0].tolist())
            d["binder"].extend([is_binder] * n_cells)
            d["clone"].extend([i] * n_cells)
            d["size_factor"].extend(size_factor)
            d["fold_increase"].extend([fold_change] * n_cells)
            d["mean"].extend(mean)
            d["concentration"].extend([concentration] * n_cells)

        if simulate_neg_control:
            adata = ad.AnnData(np.array([d["x"], d["x_neg"]], dtype="float64").T)
            adata.var_names = ["pmhc1", "neg_control"]
        else:
            adata = ad.AnnData(np.array([d["x"]]).T)
            adata.var_names = ["pmhc1"]
        adata.var["feature_types"] = ["Antigen Capture"]
        adata.obs["size_factor"] = d["size_factor"]
        adata.obs["fold_increase"] = d["fold_increase"]
        adata.obs["mean"] = d["mean"]
        adata.obs["concentration"] = d["concentration"]

        adata_tcr = ad.AnnData()
        adata_tcr.obs["is_binder"] = d["binder"]
        adata_tcr.obs["clone_id"] = d["clone"]
        if use_clonotype_cov:
            adata_tcr.uns["clone_cov"] = cov

        return md.MuData({"gex": adata, "airr": adata_tcr})

    @staticmethod
    def generate_nb_val(mu, alpha, size=1, rng_key=42):
        """Generate negative binomial samples

        Args:
            mu: the mean parameter (must be positive)
            alpha: the inverse overdispersion parameter (must be positive)
            size: the number of iid draws
            rng_key: an integer to initialize the random key generator.
        """
        return npd.NegativeBinomial2(mu, alpha).sample(jax.random.PRNGKey(rng_key), sample_shape=(size,))
