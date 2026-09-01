from typing import Union, Tuple, Any

import jax
import seaborn as sns
import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import numpyro.distributions as npd
import matplotlib.pyplot as plt

from mudata import MuData
from scipy import stats
from sklearn.metrics import precision_recall_curve


from dextrademixer.utils.utils import convert_to_invdispersion


def sample_var_from_mean(mean: Union[float, np.ndarray],
                         a: float = 2.0221541172111164, b: float = 1.6969075027280063,
                         resid_std: float = 0.31049623532404225, rng: Union[int, np.random.RandomState] = 42
                         ) -> Union[float, np.ndarray]:
    """
    Sample a realistic variance given a mean using the fitted power-law model:
        log(var) = a + b*log(mean) + Normal(0, resid_std^2)

    Args:
        mean : float or np.ndarray
            Mean(s) at which to sample the variance. Must be > 0; broadcasting allowed.
        a : float, default 2.0221541172111164
            Proportionality constant (exp(intercept) from log–log OLS).
        b : float, default 1.6969075027280063
            Scaling exponent (slope from log–log OLS).
        resid_std : float, default 0.31049623532404225
            Residual standard deviation on the *log-variance* scale (σ from OLS residuals).
        rng : int | np.random.RandomState, default 42
            Source of randomness. If int, used as the seed. If None, uses SciPy/Numpy default RNG.
    Returns:
        float or np.ndarray
            A sample of variance values with the same broadcasted shape as `mean`.
    """

    if isinstance(rng, int):
        rng = np.random.RandomState(seed=rng)
    if isinstance(mean, np.ndarray):
        size = mean.shape
    else:
        size = None
    log_var = np.log(a) + b*np.log(mean) + stats.norm(0, resid_std).rvs(size=size, random_state=rng)
    var = np.exp(log_var)

    return var


class DextramerSimulator:
    """
    Simulates dextramer single-cell data based on inferred parameters from real experiments
    """

    def __init__(self):
        self.dist_params = None

    @staticmethod
    def default_params():
        default_params = {
            'neg_mean': 2.471916508538899,
            'neg_concentration': 0.7342967361574478,
            'cells_per_clonotype': [0.2550112909684161, 2267.0, 1.0],
            'concentration_param': (0.6018940224585299, 0.09382864854673992, 3.063191246241674),
        }

        return default_params

    def simulate_pmhc_data_from_distribution(self,
                                             total_cells: int = 5000,
                                             n_clones: int = 150,
                                             binding_ratio: float = 0.05,
                                             mean_non_binder: float = None,
                                             concentration_non_binder: float = None,
                                             mean_neg_ctrl: float = None,
                                             concentration_neg_ctrl: float = None,
                                             mean_inc: float = None,
                                             var_inc: float = None,
                                             p_binding_outlier=0.0,
                                             simulate_neg_control: bool = False,
                                             plot_data: bool = False,
                                             rng_key: int = 42,
                                             rep: int = 0,
                                             ) -> Union[Tuple[MuData, Any], MuData]:
        """
        Given distribution parameters generate binding data for one pMHC. If certain parameters are not specified,
        they will be sampled from fitted distributions of real data.

        Args:
            total_cells: number of total cell to generate
            n_clones: number of clones measured in experiments.
            binding_ratio: ratio of binder vs non-binder
            mean_non_binder: mean of non-binder, if specified use this value, else sampled from fitted distribution
            concentration_non_binder: concentration parameter of non-binder, if specified use this value,
                                      else sampled from fitted distribution
            mean_neg_ctrl: mean of negative control, if specified use this value, else sampled from fitted distribution
            concentration_neg_ctrl: concentration parameter of negative control, if specified use this value,
                                    else sampled from fitted distribution
            mean_inc: fold increase of mean of binder vs non-binder, if specified use this value,
                      else sampled from fitted distribution
            var_inc: fold increase of the variance to the mean for binder NB distribution, if specified use this value,
                     else sampled from fitted distribution
            p_binding_outlier: the probability of a cell of binding clonotype to have low (noise-level) counts
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

        # No-op for reproducibility
        rng.randint(2, n_clones)

        # params
        cells_per_clonotype = params["cells_per_clonotype"]

        # Sample parameters if not provided
        if mean_neg_ctrl is None:
            mean_neg_ctrl = np.exp(stats.truncnorm(-1.0539178917389445, 1.8375518345106903, loc=1.018115879390079, scale=0.4175162931163312).rvs(random_state=rng))
        if concentration_neg_ctrl is None:
            var_neg_ctrl = sample_var_from_mean(mean_neg_ctrl, rng=rng)
            concentration_neg_ctrl = convert_to_invdispersion(mean_neg_ctrl, var_neg_ctrl)
        if mean_non_binder is None:
            mean_non_binder = np.exp(stats.truncnorm(-1.4325807532116341, 1.9485510504360735, loc=2.0461540382126118, scale=0.6019089551720753).rvs(random_state=rng))
        if concentration_non_binder is None:
            var_non_binder = sample_var_from_mean(mean_non_binder, rng=rng)
            concentration_non_binder = convert_to_invdispersion(mean_non_binder, var_non_binder)
        if mean_inc is None:
            mean_inc = stats.uniform(50, 450).rvs(random_state=rng)  # between [50, 450+50]
        mean_pos = mean_inc * mean_non_binder
        if var_inc is None:
            var_pos = sample_var_from_mean(mean_pos, rng=rng)
        else:
            var_pos = var_inc * mean_non_binder
        concentration_pos = convert_to_invdispersion(mean_pos, var_pos)

        # Sample binder assignments and cells per clone until empirical binding ratio is close to target
        max_trials = 20
        best_err = 10000
        for _ in range(max_trials):
            total_le = total_cells - n_clones
            raw_cells_per_clone = stats.boltzmann.rvs(*cells_per_clonotype, size=n_clones, random_state=rng)
            cells_per_clone_p = raw_cells_per_clone / raw_cells_per_clone.sum()
            cells_per_clone_trial = (rng.multinomial(total_le, cells_per_clone_p) + np.ones(n_clones)).astype("int32")

            # Sample multiple binder assignments and pick the one that gives empirical binding ratio closest to target
            binder_assignment_trial = rng.binomial(1, binding_ratio, size=(10000, n_clones))
            empirical_binding_ratio = ((cells_per_clone_trial * binder_assignment_trial).sum(1) / total_cells)
            # mean of error from empirical cell and clone level binder ratio
            err = ((np.abs(empirical_binding_ratio - binding_ratio) +
                   np.abs(binder_assignment_trial.mean(1) - binding_ratio))
                   / 2)

            if err.min() < best_err:
                best_idx = err.argmin()
                binder_assignment = binder_assignment_trial[best_idx]
                cells_per_clone = cells_per_clone_trial

            if err.min() < binding_ratio * 0.05:
                break

        # generate cell per clonotype following a discrete exponentially decreasing distribution normalized to
        # specified total cell count

        d = {"x": [], "binder": [], "clone": [], "fold_increase": [], "outlier":[]}
        if simulate_neg_control:
            d["x_neg"] = []

        key = jax.random.PRNGKey(rng_key)  # set starting rng_key
        for i in range(n_clones):
            # Propagate the key to create new subkeys for each clone, else the same distribution will always be sampled
            key, subkey = jax.random.split(key)

            is_binder = int(binder_assignment[i])
            n_cells = cells_per_clone[i]

            if is_binder:
                mean = mean_pos
                concentration = concentration_pos
            else:
                mean = mean_non_binder
                # add some noise to neg_concentration
                a = (0.001 - concentration_non_binder) / (concentration_non_binder / 3)
                concentration = stats.truncnorm.rvs(a, np.inf, loc=concentration_non_binder,
                                                    scale=concentration_non_binder / 3, random_state=rng)

            x = DextramerSimulator.generate_nb_val(mean, concentration, size=n_cells, rng_key=key)

            if simulate_neg_control:
                key, subkey = jax.random.split(key)
                x_neg = DextramerSimulator.generate_nb_val(mean_neg_ctrl, concentration_neg_ctrl, size=n_cells, rng_key=key)
                d["x_neg"].extend(x_neg.tolist())

            d["x"].extend(x.tolist())
            d["binder"].extend([is_binder] * n_cells)
            d["clone"].extend([i] * n_cells)
            d["fold_increase"].extend([mean_inc] * n_cells)

        if p_binding_outlier > 0:
            outlier = np.zeros(total_cells, dtype=int)
            binder_mask = np.array(d["binder"], dtype=bool)
            n_binder = binder_mask.sum()

            binder_outlier_trial = rng.binomial(1, p_binding_outlier, size=(10000, n_binder))

            err = np.abs(p_binding_outlier - binder_outlier_trial.mean(1))
            best_idx = err.argmin()
            binder_outlier = binder_outlier_trial[best_idx]
            outlier[binder_mask] = binder_outlier

            a = (0.001 - concentration_non_binder) / (concentration_non_binder / 3)
            concentration = stats.truncnorm.rvs(a, np.inf, loc=concentration_non_binder,
                                                scale=concentration_non_binder / 3, random_state=rng)
            x = np.array(d["x"])
            x[outlier.astype(bool)] = (
                DextramerSimulator.generate_nb_val(mean_non_binder, concentration, size=np.sum(outlier), rng_key=key)
            )
            d["x"] = x.tolist()
            d["outlier"] = outlier.tolist()
        else:
            d["outlier"] = [0]*total_cells

        mdat = DextramerSimulator.__generate_mdata(d, simulate_neg_control)
        # Best theoretical F1
        precision, recall, thresholds = precision_recall_curve(mdat['airr'].obs['is_binder'], mdat['gex'].X[:, 0])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]

        sim_params = {
            'mean_non_binder': mean_non_binder,
            'concentration_non_binder': concentration_non_binder,
            'mean_neg_ctrl': mean_neg_ctrl,
            'concentration_neg_ctrl': concentration_neg_ctrl,
            'mean_inc': mean_inc,
            'var_inc': var_inc,
            'mean_pos': mean_pos,
            'var_pos': var_pos,
            'concentration_pos': concentration_pos,
            'total_cells': total_cells,
            'n_clones': n_clones,
            'binding_ratio': binding_ratio,
            'p_binding_outlier': p_binding_outlier,
            'rng_key': rng_key,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'rep': rep,
        }
        mdat['gex'].uns['sim_params'] = sim_params

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
        plt.title(f'Linear (outlier thr in red)')
        i += 1

        plt.subplot(n_rows, n_cols, i)
        ax = sns.histplot(x=x_no_outliers, discrete=True, stat='percent', element='step', hue=hue_no_outliers, hue_order=['non-binder', 'binder', 'Neg control'])
        sns.despine()
        plt.title(f'Linear no outliers (z-score > 3)')
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
        plt.title(f'Log-scale (outlier thr in red)')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x_no_outliers, discrete=True, stat='percent', element='step', hue=hue_no_outliers, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title(f'Log-scale no outliers (z-score > 3)')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', binrange=(0, 100), element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title(f'Log-scale x < 100')
        sns.despine()
        i += 1

        return fig

    @staticmethod
    def __generate_mdata(d, simulate_neg_control) -> MuData:

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
        adata_tcr.obs["outlier"] = d["outlier"]

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
