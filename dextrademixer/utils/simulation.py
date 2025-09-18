import collections.abc
import numbers
import warnings
from collections import defaultdict
from typing import Union, Tuple, Optional, Any

import jax
import scipy.special

import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import mudata as md
import numpyro as npy
import numpyro.distributions as npd
import statsmodels.formula.api as smf

from mudata import MuData
from scipy import stats
from scipy.special import expit

import matplotlib.pyplot as plt

from dextrademixer.utils.utils import remove_outliers, convert_neg_binom_params, \
    convert_to_invdispersion, convert_to_variance, dist_to_sim, generate_sim_from_ltridist, \
    normalize_distance_matrix


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

    rng = np.random.RandomState(seed=rng_key)
    d = {"avidity": [], "binder": [], "clone": []}
    d_neg = {"avidity": [], "binder": [], "clone": []}
    binder_assignment = rng.binomial(1, binding_ratio, size=n_clones)

    key = jax.random.PRNGKey(rng_key)  # set starting rng_key
    for i in range(n_clones):
        key, subkey = jax.random.split(key)
        is_binder = binder_assignment[i]

        if is_binder:
            n_cell = rng.randint(*n_cells_per_binder, size=1)[0]
            mean = rng.uniform(*mean_binder_range, size=1)[0]
            shape = rng.uniform(*shape_binder_range, size=1)[0]
            d["avidity"].extend(DextramerSimulator.generate_nb_val(mean, shape, size=n_cell, rng_key=key).tolist())

        else:
            n_cell = rng.randint(*n_cells_per_non_binder, size=1)[0]
            d["avidity"].extend(
                DextramerSimulator.generate_nb_val(mean_non_binder, shape_non_binder, size=n_cell,
                                                   rng_key=key).tolist())

        d["binder"].extend([is_binder] * n_cell)
        d["clone"].extend([i] * n_cell)

        d_neg["avidity"].extend(DextramerSimulator.generate_nb_val(mean_non_binder,
                                                                   shape_non_binder, size=n_cell,
                                                                   rng_key=key).tolist())
        d_neg["binder"].extend([0] * n_cell)
        d_neg["clone"].extend([i] * n_cell)

    adata = ad.AnnData(np.array([d["avidity"], d_neg["avidity"]], dtype="float64").T)
    adata.var_names = ["pmhc1", "neg_control"]
    adata.var["feature_types"] = ["Antigen Capture", "Antigen Capture"]

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
        self.dist_params = None
        self.params = None

    @staticmethod
    def default_params():
        default_params = {
            'neg_mean': 2.471916508538899,
            'neg_concentration': 0.7342967361574478,
            'cells_per_clonotype': [0.2550112909684161, 2267.0, 1.0],
            'concentration_param': (0.6018940224585299, 0.09382864854673992, 3.063191246241674),
            'clonotype_dist_param': (1.3302992164770606, 2.1670838467235023, 0, 1),
            'lower_clonotype_dist_param': (0.6497129330485172, 0.4720738804426927, -0.008554402994886644, 0.19605440966103183),
            'upper_clonotype_dist_param': (0.8200108285624408, 7.9191977543818295, 0.4218749850980672, 1.5026569991742342)
        }

        return default_params

    def estimate_simulation_params(self,
                                   mdata: md.MuData,
                                   neg_ctrl_key: str,
                                   gex_key: str = "gex",
                                   ir_key: str = "airr",
                                   ir_dist_key: str = "dist",
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
        rng = np.random.RandomState(seed=rng_key)
        i = 0

        def __remove_extreme_values(x, is_filter, iqr):
            nonlocal i
            i += 1
            return remove_outliers(x, iqr) if is_filter else x

        if not isinstance(mdata, md.MuData):
            raise ValueError("`mdat`is not a MuData object. Please read the scirpy tutorial to combine GEX and AIRR "
                             "data.")

        if isinstance(filter_extreme_values, bool):
            filter_extreme_values = [filter_extreme_values] * 4
        if isinstance(filter_extreme_values, collections.abc.Collection) and len(filter_extreme_values) < 4:
            raise ValueError("`filter_extreme_values` must have a length of at least four.")

        if isinstance(iq_range, numbers.Number) and not isinstance(iq_range, bool):
            iq_range = [iq_range] * 4

        if isinstance(iq_range, collections.abc.Collection) and len(iq_range) < 4:
            raise ValueError("`iq_range` must have a length of at least four.")

        dist_param = {}
        param = {}

        # normalize gex data
        X = mdata.mod[gex_key].X
        neg_idx = mdata.mod[gex_key].var["gene_ids"].to_list().index(neg_ctrl_key)

        #####################
        # Estimate parameters
        #####################
        neg_x = __remove_extreme_values(X[:, neg_idx].toarray()[:, 0], filter_extreme_values[i], iq_range[i])

        # estimation of mean and inverse dispersion parameter from nb model
        nbfit = smf.negativebinomial("nbdata ~ 1",
                                     data=pd.DataFrame({"nbdata": neg_x})).fit(disp=False)

        dist_param["neg_mean"] = np.exp(nbfit.params.iloc[0])
        dist_param["neg_concentration"] = 1 / nbfit.params.iloc[1]
        param["neg_x"] = neg_x

        # fit clonotype size distribution
        clone_size = mdata.mod[ir_key].obs.groupby("clone_id", dropna=False).size()
        rv = stats.boltzmann
        bounds = [(0, 10000), (1, np.max(clone_size)), (1, 1)]
        clone_size = __remove_extreme_values(clone_size, filter_extreme_values[i], iq_range[i])
        res = stats.fit(rv, clone_size, bounds)

        if not res.success:
            warnings.warn("Estimation of boltzmann parameters of clone sizes failed. Please "
                          "adjust boundary conditions of the parameters")

        dist_param["cells_per_clonotype"] = list(res.params)
        param["cells_per_clonotype"] = clone_size

        # fit inv dispersion distribution
        invdisp = []
        var = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for c, g in mdata.mod[ir_key].obs.groupby("clone_id", dropna=False):
                if g.shape[0] < 15:  #at least 15 cells to fit neg_binom model
                    continue
                m = mdata.mod["gex"][g.index]
                for j in m.var.gene_ids:
                    d = m[:, j].to_df()
                    d = d.rename({j: j.replace("-", "_")}, axis=1)
                    nbfit = smf.negativebinomial(f"{d.columns[0]} ~ 1", data=d, loglike_method="nb2").fit(disp=False)
                    if not nbfit.converged:
                        continue
                    invdisp.append(1 / nbfit.params.iloc[1])  # concentration parameter
                    var.append(convert_to_variance(np.exp(nbfit.params.iloc[0]), nbfit.params.iloc[1]))

        invdisp = __remove_extreme_values(np.array(invdisp), filter_extreme_values[i], iq_range[i])
        dist_param["concentration_param"] = stats.gamma.fit(invdisp)
        param["concentration"] = invdisp
        param["variance"] = var

        # fit prior for covariance matrix
        dist = mdata.mod[ir_key].uns[ir_dist_key]

        cov = dist_to_sim(dist, normalize=True)

        dist_norm = normalize_distance_matrix(dist)
        dist_flat = dist_norm[np.triu_indices_from(dist_norm, k=-1)] # lower-triangle without diagnoal
        dist_flat = np.clip(dist_flat, 1e-6, 1-1e-6)
        perc_10, perc_50 = np.percentile(dist_flat, [10, 50])

        dist_param["clonotype_dist_param"] = stats.beta.fit(dist_flat, floc=0, fscale=1)
        dist_param["lower_clonotype_dist_param"] = stats.beta.fit(dist_flat[dist_flat <= perc_10])
        dist_param["upper_clonotype_dist_param"] = stats.beta.fit(dist_flat[dist_flat >= perc_50])
        param["clonotype_dist"] = dist_flat

        self.dist_params = dist_param
        self.params = param

        # QC plot
        if plot_qc:
            return self.__qc_plot(neg_x, clone_size, invdisp, dist_norm, cov, dist_flat, rng)

    def __qc_plot(self, neg_x, clone_size, invdisp, dist, cov, dist_flat, rng):
        """
        Plots QQ plots of fitted theoretical distribution against empirical distribution
        """

        if self.dist_params is not None:
            params = {**DextramerSimulator.default_params(), **self.dist_params}
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
        sns.histplot(stats.nbinom.rvs(*negbinom_params, size=sample_size, random_state=rng), log_scale=True,
                     legend=False, ax=axs[0, 2])
        axs[0, 2].title.set_text("Fitted negative control distribution")

        sns.histplot(clone_size, log_scale=True, legend=False, ax=axs[1, 0])
        axs[1, 0].title.set_text("Empirical clone size distribution")
        stats.probplot(clone_size, dist=stats.boltzmann,
                       sparams=params["cells_per_clonotype"], plot=axs[1, 1], rvalue=True)
        axs[1, 1].get_children()[2].set_fontsize("x-small")
        axs[1, 1].title.set_text("Discrete Boltzmann fitted clone size")
        axs[1, 1].get_lines()[0].set_color(blue)
        sns.histplot(stats.boltzmann.rvs(*params["cells_per_clonotype"], size=sample_size, random_state=rng),
                     log_scale=True, legend=False, ax=axs[1, 2])
        axs[1, 2].title.set_text("Fitted clone size distribution")

        sns.histplot(x=invdisp, log_scale=False, legend=False, ax=axs[2, 0])
        axs[2, 0].title.set_text("Empirical inverse dispersion \n distribution of clonotypes")
        stats.probplot(invdisp, dist=stats.gamma,
                       sparams=params["concentration_param"], plot=axs[2, 1], rvalue=True)
        axs[2, 1].title.set_text("Gamma fitted inverse dispersion \n of clonotypes")
        axs[2, 1].get_children()[2].set_fontsize("x-small")
        axs[2, 1].get_lines()[0].set_color(blue)
        axs[2, 2].title.set_text("Fitted inverse dispersion \n distribution of clonotypes ")
        sns.histplot(stats.gamma.rvs(*params["concentration_param"], size=sample_size, random_state=rng),
                     log_scale=False, legend=False, ax=axs[2, 2])

        axs[3, 0].title.set_text("Empirical clonotype \n distance distribution")
        sns.histplot(dist_flat, log_scale=False, ax=axs[3, 0])
        stats.probplot(dist_flat, dist="beta", sparams=params["clonotype_dist_param"], plot=axs[3, 1], rvalue=True)
        axs[3, 1].title.set_text("Beta fitted normalized distances")
        axs[3, 1].get_children()[2].set_fontsize("x-small")
        axs[3, 1].get_lines()[0].set_color(blue)
        axs[3, 2].title.set_text("Fitted distance distribution")
        sns.histplot(stats.beta.rvs(*params["clonotype_dist_param"], size=sample_size, random_state=rng),
                     log_scale=False, legend=False, ax=axs[3, 2])

        perc_10, perc_50 = np.percentile(dist_flat, [10, 50])
        axs[4, 0].title.set_text("Empirical clonotype \n distance distribution <=10th percentile")
        sns.histplot(dist_flat[dist_flat <= perc_10], log_scale=False, ax=axs[4, 0])
        stats.probplot(dist_flat[dist_flat <= perc_10], dist="beta", sparams=params["lower_clonotype_dist_param"], plot=axs[4, 1], rvalue=True)
        axs[4, 1].title.set_text("Beta fitted normalized distances")
        axs[4, 1].get_children()[2].set_fontsize("x-small")
        axs[4, 1].get_lines()[0].set_color(blue)
        sample = stats.beta.rvs(*params["lower_clonotype_dist_param"], size=sample_size, random_state=rng)
        axs[4, 2].title.set_text("Fitted distance distribution. min: {}".format(np.min(sample)))

        sns.histplot(np.clip(stats.beta.rvs(*params["lower_clonotype_dist_param"], size=sample_size, random_state=rng),0,1),
                     log_scale=False, legend=False, ax=axs[4, 2])


        axs[5, 0].title.set_text("Empirical clonotype \n distance distribution >=50th percentile")
        sns.histplot(dist_flat[dist_flat >= perc_50], log_scale=False, ax=axs[5, 0])
        stats.probplot(dist_flat[dist_flat >= perc_50], dist="beta", sparams=params["upper_clonotype_dist_param"], plot=axs[5, 1], rvalue=True)
        axs[5, 1].title.set_text("Beta fitted normalized distances")
        axs[5, 1].get_children()[2].set_fontsize("x-small")
        axs[5, 1].get_lines()[0].set_color(blue)
        axs[5, 2].title.set_text("Fitted distance distribution")
        sns.histplot(np.clip(stats.beta.rvs(*params["upper_clonotype_dist_param"], size=sample_size, random_state=rng),0,1),
                     log_scale=False, legend=False, ax=axs[5, 2])

        axs[6, 0].title.set_text("Distance matrix \n between clonotypes")
        sns.heatmap(dist, square=True, ax=axs[6, 0], cbar_kws={"shrink": 0.5})
        axs[6, 1].title.set_text("Covariance matrix \n between clonotypes")
        sns.heatmap(cov, square=True, ax=axs[6, 1], cbar_kws={"shrink": 0.5})
        c = len(clone_size)

        d = stats.beta(*params["clonotype_dist_param"]).rvs(size=int(c*(c-1)/2), random_state=rng)
        cov_est = generate_sim_from_ltridist(d)
        axs[6, 2].title.set_text("Distance simulated \n covariance matrix")
        sns.heatmap(cov_est, square=True, ax=axs[6, 2], cbar_kws={"shrink": 0.5})

        return axs

    def simulate_pmhc_data_from_distribution(self,
                                             total_cells: int = 5000,
                                             nof_clones: int = 150,
                                             binding_ratio: float = 0.05,
                                             binding_fold_increase_range: list[float] = None,
                                             variance_fold_increase_range: list[float] = None,
                                             p_nonbinding_clone_outlier=0.0,
                                             p_binding_outlier=0.0,
                                             nof_clonotype_cluster=None,
                                             use_clonotype_cov: bool = False,
                                             simulate_neg_control: bool = False,
                                             plot_data: bool = False,
                                             rng_key: int = 42
                                             ) -> Union[Tuple[MuData, Any], MuData]:
        """
        Given negative control mean and concentration parameters (estimated from real data) generate binding data for
        one pMHC with predefined positive fold-change.

        Args:
            total_cells: number of total cell to generate
            nof_clones: number of clones measured in experiments.
            binding_ratio: ratio of binder vs non-binder
            binding_fold_increase_range: list of fold increase for pMHC binding cells
            variance_fold_increase_range: list of fold increase of the variance to the mean of a negative binomial
                                          (i.e. variance_fold_increase_range=[1] => mean = var).
                                          If not specified estimated inverdispersion of clonotypes will be used
            p_nonbinding_clone_outlier: The probability that a non-binding clone is clustered together with binding clones
            p_binding_outlier: the probability of a cell of binding clonotype to have low pMHC counts
            nof_clonotype_cluster: number of clonotype clusters to simulate in case covariance matrix should be
                                   simulated (if None than randomly sampled between [2, nof_clones]
            use_clonotype_cov: whether to use clonotype covariance to assign binding or randomly (default: False)
            simulate_neg_control: whether to simulate a negative control pMHC for each cell (default: False)
            plot_data: boolean whether to plot simulated data (default: False)
            rng_key: random seed.

        Returns:
            An Anndata object containing all generated count data and clonal information, and binder status
        """
        rng = np.random.RandomState(seed=rng_key)

        if self.dist_params is not None:
            params = {**DextramerSimulator.default_params(), **self.dist_params}
        else:
            params = DextramerSimulator.default_params()

        if variance_fold_increase_range is not None and any(v <= 1 for v in variance_fold_increase_range):
            raise ValueError("`variance_fold_increase_range` contains fold increases <= 1. " +
                             "Fold increases must be > 1")

        if nof_clonotype_cluster is not None:
            if nof_clonotype_cluster > nof_clones:
                raise ValueError("`nof_clonotype_cluster` must be smaller than `nof_clones`")
            if nof_clonotype_cluster < 2:
                raise ValueError("`nof_clonotype_cluster` must be at least 2")
        else:
            nof_clonotype_cluster = rng.randint(2, nof_clones)

        # params
        neg_mean = params["neg_mean"]
        neg_concentration = params["neg_concentration"]
        cells_per_clonotype = params["cells_per_clonotype"]
        concentration_param = params["concentration_param"]

        if binding_fold_increase_range is None:
            binding_fold_increase_range = [2, 5, 10, 50, 100, 150, 200, 500]

        binder_assignment = rng.binomial(1, binding_ratio, size=nof_clones)
        K = None
        cc_assignment = None

        # simulate TCR similarity clusters
        if use_clonotype_cov:

            cc_assignment = self.__cc_assignment(binder_assignment,
                                                       nof_clones,
                                                       nof_clonotype_cluster,
                                                       p_nonbinding_clone_outlier, rng)

            K = self.__construct_tcr_kernel(nof_clones, cc_assignment, params, rng)


        # generate cell per clonotype following a discrete exponentially decreasing distribution normalized to
        # specified total cell count
        total_le = total_cells - nof_clones
        raw_cells_per_clone = np.array([stats.boltzmann.rvs(*cells_per_clonotype,random_state=rng) for _ in range(nof_clones)])
        cells_per_clone_p = raw_cells_per_clone/raw_cells_per_clone.sum()
        cells_per_clone = (rng.multinomial(total_le, cells_per_clone_p) + np.ones(nof_clones)).astype("int32")

        d = {"x": [], "binder": [], "clone": [], "fold_increase": [], "outlier":[]}
        if simulate_neg_control:
            d["x_neg"] = []
        key = jax.random.PRNGKey(rng_key)  # set starting rng_key
        for i in range(nof_clones):
            # Propagate the key to create new subkeys for each clone, else the same distribution will always be sampled
            key, subkey = jax.random.split(key)

            is_binder = int(binder_assignment[i])
            n_cells = cells_per_clone[i]

            if is_binder:
                fold_change = float(rng.choice(binding_fold_increase_range))
                mean = fold_change * neg_mean
                if variance_fold_increase_range is None:
                    concentration = stats.gamma.rvs(*concentration_param, random_state=rng)
                else:
                    concentration = convert_to_invdispersion(mean, mean*rng.choice(variance_fold_increase_range))
            else:
                fold_change = 0.0
                mean = neg_mean
                # add some noise to neg_concentration
                a = (0.001 - neg_concentration) / (neg_concentration / 3)
                concentration = stats.truncnorm.rvs(a, np.inf, loc=neg_concentration, scale=neg_concentration / 3,
                                                    random_state=rng)

            x = DextramerSimulator.generate_nb_val(mean, concentration, size=n_cells, rng_key=key)
            if p_binding_outlier > 0 and is_binder:
                outlier = stats.binom.rvs(1, p_binding_outlier, size=n_cells, random_state=rng)
                outlier_idx = np.where(outlier)

                a = (0.001 - neg_concentration) / (neg_concentration / 3)
                concentration = stats.truncnorm.rvs(a, np.inf, loc=neg_concentration, scale=neg_concentration / 3,
                                                    random_state=rng)

                x = x.at[outlier_idx].set(
                    DextramerSimulator.generate_nb_val(mean, concentration, size=np.sum(outlier), rng_key=key)
                )
                d["outlier"].extend(outlier.tolist())
            else:
                d["outlier"].extend([0]*n_cells)

            if simulate_neg_control:
                key, subkey = jax.random.split(key)
                mean = neg_mean
                x_neg = DextramerSimulator.generate_nb_val(mean, concentration, size=n_cells, rng_key=key)
                d["x_neg"].extend(x_neg.tolist())

            d["x"].extend(x.tolist())
            d["binder"].extend([is_binder] * n_cells)
            d["clone"].extend([i] * n_cells)
            d["fold_increase"].extend([fold_change] * n_cells)

        mdat = DextramerSimulator.__generate_mdata(d, simulate_neg_control, K, cc_assignment)
        if plot_data:
            return mdat, DextramerSimulator.__plot_simulated_data(d)
        else:
            return mdat

    def simulate_pmhc_data_from_sample(self,
                                       total_cells: int = 5000,
                                       nof_clones: int = 150,
                                       binding_ratio: float = 0.05,
                                       binding_fold_increase_range: list[float] = None,
                                       use_clonotype_cov: bool = False,
                                       nof_clonotype_cluster = None,
                                       p_nonbinding_clone_outlier = 0.0,
                                       simulate_neg_control: bool = False,
                                       plot_data: bool = False,
                                       rng_key: int = 42
                                       ) -> Union[Tuple[MuData, Any], MuData]:
        """
        Given negative control samples and other parameters sampled from real world data, generate binding data for
        one pMHC with predefined positive fold-change.

        Args:
            total_cells: number of total cell to generate
            nof_clones: number of clones measured in experiments.
            binding_ratio: ratio of binder vs non-binder
            binding_fold_increase_range: list of fold increase for pMHC binding cells
            use_clonotype_cov: whether to use clonotype covariance to assign binding or randomly (default: False)
            simulate_neg_control: whether to simulate a negative control pMHC for each cell (default: False)
            plot_data: boolean whether to plot simulated data (default: False)
            rng_key: random seed.

        Returns:
            An Anndata object containing all generated count data and clonal information, and binder status
        """

        if self.params is None:
            raise RuntimeError("Please estimate real world parameters with `estimate_simulation_params`.")

        rng = np.random.RandomState(seed=rng_key)

        # params
        neg_x = self.params["neg_x"]
        cells_per_clonotype = self.params["cells_per_clonotype"]

        if binding_fold_increase_range is None:
            binding_fold_increase_range = [2, 5, 10, 50, 100, 150, 200, 500]

        if nof_clonotype_cluster is not None:
            if nof_clonotype_cluster > nof_clones:
                raise ValueError("`nof_clonotype_cluster` must be smaller than `nof_clones`")
            if nof_clonotype_cluster < 2:
                raise ValueError("`nof_clonotype_cluster` must be at least 2")
        else:
            nof_clonotype_cluster = rng.randint(2, nof_clones)

        d = {"x": [], "binder": [], "clone": [], "fold_increase": []}
        if simulate_neg_control:
            d["x_neg"] = []

        binder_assignment = rng.binomial(1, binding_ratio, size=nof_clones)
        K = None
        cc_assignment = None

        # simulate TCR similarity clusters
        if use_clonotype_cov:
            cc_assignment = self.__cc_assignment(binder_assignment,
                                                 nof_clones,
                                                 nof_clonotype_cluster,
                                                 p_nonbinding_clone_outlier, rng)

            K = self.__construct_tcr_kernel(nof_clones, cc_assignment, self.dist_params, rng)

        # generate cell per clonotype following a discrete exponentially decreasing distribution normalized to
        # specified total cell count
        total_le = total_cells - nof_clones
        raw_cells_per_clone = rng.choice(cells_per_clonotype, size=nof_clones)
        cells_per_clone_p = stats.dirichlet.rvs(raw_cells_per_clone, random_state=rng)[0]
        cells_per_clone = (rng.multinomial(total_le, cells_per_clone_p) + np.ones(nof_clones)).astype("int32")

        for i in range(nof_clones):
            is_binder = binder_assignment[i]
            n_cells = cells_per_clone[i]
            fold_change = rng.choice(binding_fold_increase_range)
            nx = rng.choice(neg_x, size=n_cells)
            x = fold_change*nx if is_binder else nx

            if simulate_neg_control:
                d["x_neg"].extend(rng.choice(neg_x, size=n_cells).tolist())

            d["x"].extend(x.tolist())
            d["binder"].extend([is_binder] * n_cells)
            d["clone"].extend([i] * n_cells)
            d["fold_increase"].extend([fold_change] * n_cells)

        mdat = DextramerSimulator.__generate_mdata(d, simulate_neg_control, K, cc_assignment)

        if plot_data:
            return mdat, DextramerSimulator.__plot_simulated_data(d)
        else:
            return mdat

    @staticmethod
    def __plot_simulated_data(d):
        df = pd.DataFrame.from_dict(d)

        x = df["x"].values.reshape(-1, )
        hue = pd.Series(df["binder"]).map({0: "non-binder", 1: "binder"})
        x_log = np.log(x + 1)  # Transform to log scale, roughly normal distributed
        zscore = (x_log - x_log.mean()) / x_log.std()
        x_no_outliers = x[zscore < 4]
        hue_no_outliers = hue[zscore < 4]
        outlier_thr = x_no_outliers.max()

        if "x_neg" in df:
            x_neg = df["x_neg"].values.reshape(-1, )
            x = np.concatenate((x, x_neg), axis=0)
            x_no_outliers = np.concatenate((x_no_outliers, x_neg), axis=0)
            hue = pd.concat([hue, pd.Series(["Neg control"]*len(x_neg))], axis=0)
            hue_no_outliers = pd.concat([hue_no_outliers, pd.Series(["Neg control"]*len(x_neg))], axis=0)

        n_cols = 3
        n_rows = 2
        fig = plt.figure(figsize=(3 * n_cols, 2.4 * n_rows))
        i = 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.axvline(outlier_thr, ls='--', c='r')
        sns.despine()
        plt.title(f'Linear (outlier threshold in red)')
        i += 1

        plt.subplot(n_rows, n_cols, i)
        ax = sns.histplot(x=x_no_outliers, discrete=True, stat='percent', element='step', hue=hue_no_outliers, hue_order=['non-binder', 'binder', 'Neg control'])
        sns.despine()
        plt.title(f'Linear no outliers (log-transformed z-score > 3)')
        i += 1

        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=False, title='Binding status')

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', binrange=(0, 100), element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.title(f'Linear x < 100')
        sns.despine()
        i += 1

        # Log scale
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.axvline(outlier_thr, ls='--', c='r')
        plt.title(f'Log-scale (outlier threshold in red)')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x_no_outliers, discrete=True, stat='percent', element='step', hue=hue_no_outliers, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title(f'Log-scale no outliers (log-transformed z-score > 3)')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', binrange=(0, 100), element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title(f'Log-scale x < 100')
        sns.despine()
        i += 1

        plt.show()

        return fig

    @staticmethod
    def __generate_mdata(d, simulate_neg_control, cov, cc_assignment) -> MuData:

        if simulate_neg_control:
            adata = ad.AnnData(np.array([d["x"], d["x_neg"]], dtype="int64").T)
            adata.var_names = ["pmhc1", "neg_control"]
            adata.var["feature_types"] = ["Antigen Capture", "Antigen Capture"]
        else:
            adata = ad.AnnData(np.array([d["x"]], dtype="int64").T)
            adata.var_names = ["pmhc1"]
            adata.var["feature_types"] = ["Antigen Capture"]

        adata.obs["fold_increase"] = d["fold_increase"]
        adata.obs.index = adata.obs.index.astype("int32")

        adata_tcr = ad.AnnData()
        adata_tcr.obs["is_binder"] = d["binder"]
        adata_tcr.obs["clone_id"] = d["clone"]

        if cov is not None:
            adata_tcr.obs["cc_aa_sim"] = cc_assignment[d["clone"]]
            cc_size = np.bincount(adata_tcr.obs["cc_aa_sim"])
            adata_tcr.obs["cc_aa_sim_size"] = cc_size[adata_tcr.obs["cc_aa_sim"]]
            adata_tcr.uns["clone_cov"] = np.array(cov)

        return md.MuData({"gex": adata, "airr": adata_tcr})

    @staticmethod
    def generate_nb_val(mu, alpha, size=1, rng_key=42):
        """Generate negative binomial samples

        Args:
            mu: the mean parameter (must be positive)
            alpha: the inverse overdispersion parameter (must be positive)
            size: the number of iid draws
            rng_key: int or jax.random.PRNGKey as random seed
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)
        return npd.NegativeBinomial2(mu, alpha).sample(rng_key, sample_shape=(size,))

    @staticmethod
    def __cc_assignment(binder_assignment, nof_clones, nof_clonotype_cluster, p_nonbinding_clone_outlier, rng):
        """
        Split clonotypes into clusters but ensure perfect separation of binding assignment.
        Missassigns with `p_nonbinding_clone_outlier`probability nonbinding clones to binding clusters

        Returns: an array with clonotype cluster assignments
        """

        def randomly_assign_to_clusters(assignments, indices, start_cluster, n_clusters):
            # Randomly assign each sample to a cluster
            cluster_choices = np.arange(start_cluster, start_cluster + n_clusters)

            # Ensure at least one sample per cluster
            if len(indices) >= n_clusters:
                initial_assignments = np.random.choice(indices, n_clusters, replace=False)
                for i, idx in enumerate(initial_assignments):
                    assignments[idx] = start_cluster + i

                # Then randomly assign remaining samples
                remaining_indices = np.setdiff1d(indices, initial_assignments)
                if len(remaining_indices) > 0:
                    random_clusters = np.random.choice(cluster_choices, size=len(remaining_indices))
                    assignments[remaining_indices] = random_clusters
            else:
                random_clusters = np.random.choice(cluster_choices, size=len(indices))
                assignments[indices] = random_clusters

            return assignments

        # make local copy as we might modify the assignment to infuse errors
        binder_assignment = np.asarray(binder_assignment).copy()

        # Inject errors into cc_assignments via label switching of non-binders
        if 0 < p_nonbinding_clone_outlier < 1:
            nonbinder_indices = np.where(binder_assignment == 0)[0]
            n_nonbinder = len(nonbinder_indices)

            n_errors = rng.binomial(n_nonbinder, p_nonbinding_clone_outlier)
            if n_errors > 0:
                error_indices = rng.choice(nonbinder_indices, size=n_errors, replace=False)
                binder_assignment[error_indices] = 1

        nonbinder_indices = np.where(binder_assignment == 0)[0]
        binder_indices = np.where(binder_assignment == 1)[0]

        # Randomly assign number of clusters to each class
        n_clusters_nonbinder = rng.randint(1, nof_clonotype_cluster)
        n_clusters_binder = nof_clonotype_cluster - n_clusters_nonbinder

        # Initialize cluster assignments
        cluster_assignments = randomly_assign_to_clusters(np.zeros(nof_clones, dtype=int), nonbinder_indices, 0,
                                                          n_clusters_nonbinder)
        cluster_assignments = randomly_assign_to_clusters(cluster_assignments, binder_indices, n_clusters_nonbinder,
                                                          n_clusters_binder)

        return cluster_assignments

    @staticmethod
    def __construct_tcr_kernel(n_clones, cc_assignment, params, rng):
        """
            construct the TCR-similarity Kernel based on clonotype cluster assignments.

            Returns: an n_clones x n_clones similarity matrix
        """

        def tril_indices_from_subset(row_idx, col_idx):
            """
            Get strictly lower triangular indices for a subset of indices in an n x n matrix.

            Parameters:
                row_idx (array-like): Selected subset of row indices.
                col_idx (array-like): Selected subset of colume indices.

            Returns:
                tuple: (row_indices, col_indices) for strictly lower triangular elements.
            """
            r_grid, c_grid = np.meshgrid(row_idx, col_idx, indexing='ij')
            mask = r_grid > c_grid
            return r_grid[mask], c_grid[mask]

        # first extra inter and intra distance parameters
        inter_dist_param = list(params["upper_clonotype_dist_param"])
        intra_dist_param = list(params["lower_clonotype_dist_param"])

        c_ids = np.arange(n_clones)

        cc_to_clone = defaultdict(list)
        for c, cc in enumerate(cc_assignment):
            cc_to_clone[cc].append(c)

        # initialize Kernel with only inter distances
        K = np.zeros([n_clones,n_clones])

        # iterate through cc simulate intra distances and replace values in K
        for cc, c_idx in cc_to_clone.items():
            n_cc = len(c_idx)
            c_idx = np.array(c_idx)
            K_intra = np.clip(stats.beta.rvs(*intra_dist_param, size=int(n_cc*(n_cc-1)/2),
                                           random_state=rng),0,1)
            sub_tr_row, sub_tr_col = tril_indices_from_subset(c_idx, c_idx)
            K[(sub_tr_row, sub_tr_col)] = K_intra

            # generate inter-clonal distance while shifting also the mean of the distribution randomly
            # get clonotype not contained in current clone cluster
            c_inter  = np.setdiff1d(c_ids, c_idx)
            sub_tr_row, sub_tr_col = tril_indices_from_subset(c_idx, c_inter)

            #shift mean
            tmp = np.copy(inter_dist_param)
            tmp[2] = inter_dist_param[2] + rng.uniform(-0.2, 0.3, size=1)
            K_inter = np.clip(stats.beta.rvs(*(tmp), size=len(sub_tr_row),
                                            random_state=rng), 0, 0.9)
            K[(sub_tr_row, sub_tr_col)] = K_inter

        # set diagonal to 0
        K += np.tril(K).T
        K[np.diag_indices(n_clones)] = 0
        return dist_to_sim(jax.numpy.array(K), normalize=False, epsilon=1e-6)
