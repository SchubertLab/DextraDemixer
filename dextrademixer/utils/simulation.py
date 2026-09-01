from typing import Union, Tuple, Optional, Any

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

Z_SCORE_CUTOFF = 4


class DextramerSimulator:
    """
    Simulates dextramer single-cell data based on inferred parameters from real experiments
    """

    @staticmethod
    def default_params():
        """
        Parameters of the distributions the simulator samples from when the corresponding
        argument of `simulate_pmhc_data_from_distribution` is left unspecified. All values
        were fitted on 10x Genomics pMHC-multimer data

        Returns:
            A dict of distribution parameters, in the argument order of the corresponding
            `scipy.stats` distribution.
        """
        default_params = {
            # m_c ~ Boltzmann(lambda, N, loc) -- raw clone size
            'clone_size_boltzmann': [0.2550112909684161, 2267.0, 1.0],
            # mu_0 ~ exp(truncnorm(a, b, loc, scale)) -- noise mean
            'mean_noise_trunc_lognorm': (-1.4325807532116341, 1.9485510504360735,
                                         2.0461540382126118, 0.6019089551720753),
            # mu_neg ~ exp(truncnorm(a, b, loc, scale)) -- negative control mean
            'mean_neg_ctrl_trunc_lognorm': (-1.0539178917389445, 1.8375518345106903,
                                            1.018115879390079, 0.4175162931163312),
            # beta ~ Uniform(loc, scale), i.e. [50, 500] -- signal-to-noise ratio
            'signal_to_noise_uniform': (50, 450),
            # ln(sigma^2) = ln(a) + b*ln(mu) + Normal(0, resid_std^2) -- mean-variance relation
            'mean_var_powerlaw': (2.0221541172111164, 1.6969075027280063, 0.31049623532404225),
        }

        return default_params

    @staticmethod
    def simulate_pmhc_data_from_distribution(total_cells: int = 5000,
                                             n_clones: int = 150,
                                             binder_fraction: float = 0.05,
                                             mean_noise: Optional[float] = None,
                                             mean_neg_ctrl: Optional[float] = None,
                                             inv_dispersion_noise: Optional[float] = None,
                                             inv_dispersion_neg_ctrl: Optional[float] = None,
                                             signal_to_noise: Optional[float] = None,
                                             signal_var_ratio: Optional[float] = None,
                                             p_dropout: float = 0.0,
                                             simulate_neg_control: bool = False,
                                             plot_data: bool = False,
                                             rng_key: int = 42,
                                             rep: int = 0,
                                             ) -> Union[Tuple[MuData, Any], MuData]:
        """
        Given distribution parameters generate binding data for one pMHC. If certain parameters are not specified,
        they will be sampled from fitted distributions of real data.


        Args:
            total_cells: number of cells to generate (N)
            n_clones: number of clones measured in experiments (C)
            binder_fraction: fraction of clones that are antigen-specific binders (p)
            mean_noise: mean of the noise component, if specified use this value,
                        else sampled from fitted distribution (mu_0)
            mean_neg_ctrl: mean of the negative control, if specified use this value,
                           else sampled from fitted distribution (mu_neg)
            inv_dispersion_noise: inverse dispersion of the noise component, if specified use this value,
                                  else derived from the sampled variance (alpha_0)
            inv_dispersion_neg_ctrl: inverse dispersion of the negative control, if specified use this value,
                                     else derived from the sampled variance (alpha_neg)
            signal_to_noise: ratio of the signal to the noise component mean, if specified use this value,
                             else sampled from fitted distribution (beta, with mu_1 = beta * mu_0)
            signal_var_ratio: variance of the signal component as a multiple of its mean, if specified use this
                              value, else sampled from the fitted mean-variance relation
            p_dropout: probability that a cell of a binding clonotype drops out, i.e. carries noise-level
                       counts instead (p_out)
            simulate_neg_control: whether to simulate a negative control pMHC for each cell (default: False)
            plot_data: boolean whether to plot simulated data (default: False)
            rng_key: random seed.
            rep: replicate index, stored in `sim_params` only.

        Returns:
            A MuData with a 'gex' modality holding the simulated counts and a 'airr' modality
            holding clone ids and binder status. If `plot_data`, a (MuData, Figure) tuple.
        """
        rng = np.random.RandomState(seed=rng_key)
        params = DextramerSimulator.default_params()

        # No-op for reproducibility
        rng.randint(2, n_clones)

        # params
        clone_size_boltzmann = params["clone_size_boltzmann"]
        mean_noise_trunc_lognorm = params["mean_noise_trunc_lognorm"]
        mean_neg_ctrl_trunc_lognorm = params["mean_neg_ctrl_trunc_lognorm"]
        signal_to_noise_uniform = params["signal_to_noise_uniform"]

        # Sample parameters if not provided
        if mean_neg_ctrl is None:
            a, b, loc, scale = mean_neg_ctrl_trunc_lognorm
            mean_neg_ctrl = np.exp(stats.truncnorm(a, b, loc=loc, scale=scale).rvs(random_state=rng))
        if inv_dispersion_neg_ctrl is None:
            var_neg_ctrl = DextramerSimulator.sample_var_from_mean(mean_neg_ctrl, rng=rng)
            inv_dispersion_neg_ctrl = convert_to_invdispersion(mean_neg_ctrl, var_neg_ctrl)
        if mean_noise is None:
            a, b, loc, scale = mean_noise_trunc_lognorm
            mean_noise = np.exp(stats.truncnorm(a, b, loc=loc, scale=scale).rvs(random_state=rng))
        if inv_dispersion_noise is None:
            var_noise = DextramerSimulator.sample_var_from_mean(mean_noise, rng=rng)
            inv_dispersion_noise = convert_to_invdispersion(mean_noise, var_noise)
        if signal_to_noise is None:
            signal_to_noise = stats.uniform(*signal_to_noise_uniform).rvs(random_state=rng)
        mean_signal = signal_to_noise * mean_noise
        if signal_var_ratio is None:
            var_signal = DextramerSimulator.sample_var_from_mean(mean_signal, rng=rng)
        else:
            var_signal = signal_var_ratio * mean_signal
        inv_dispersion_signal = convert_to_invdispersion(mean_signal, var_signal)

        # Sample binding labels and clone sizes until the empirical binder fraction is close to the target
        max_trials = 20
        best_err = 10000
        for _ in range(max_trials):
            cells_to_distribute = total_cells - n_clones
            raw_clone_size = stats.boltzmann.rvs(*clone_size_boltzmann, size=n_clones, random_state=rng)
            clone_size_proportion = raw_clone_size / raw_clone_size.sum()
            clone_size_trial = (rng.multinomial(cells_to_distribute, clone_size_proportion)
                                + np.ones(n_clones)).astype("int32")

            # Sample multiple binding labels and pick the one closest to the target binder fraction
            binder_label_trial = rng.binomial(1, binder_fraction, size=(10000, n_clones))
            empirical_binder_fraction = ((clone_size_trial * binder_label_trial).sum(1) / total_cells)
            # mean of error from empirical cell and clone level binder fraction
            err = ((np.abs(empirical_binder_fraction - binder_fraction) +
                   np.abs(binder_label_trial.mean(1) - binder_fraction))
                   / 2)

            if err.min() < best_err:
                best_idx = err.argmin()
                binder_label = binder_label_trial[best_idx]
                clone_size = clone_size_trial

            if err.min() < binder_fraction * 0.05:
                break

        # generate cell per clonotype following a discrete exponentially decreasing distribution normalized to
        # specified total cell count

        d = {"x": [], "is_binder": [], "clone_id": [], "signal_to_noise": [], "dropout": []}
        if simulate_neg_control:
            d["x_neg"] = []

        key = jax.random.PRNGKey(rng_key)  # set starting rng_key
        for i in range(n_clones):
            # Propagate the key to create new subkeys for each clone, else the same distribution will always be sampled
            key, subkey = jax.random.split(key)

            is_binder = int(binder_label[i])
            n_cells = clone_size[i]

            if is_binder:
                clone_mean = mean_signal
                clone_inv_dispersion = inv_dispersion_signal
            else:
                clone_mean = mean_noise
                # add some noise to the noise component's inverse dispersion
                trunc_lower_bound = (0.001 - inv_dispersion_noise) / (inv_dispersion_noise / 3)
                clone_inv_dispersion = stats.truncnorm.rvs(trunc_lower_bound, np.inf, loc=inv_dispersion_noise,
                                                           scale=inv_dispersion_noise / 3, random_state=rng)

            x = DextramerSimulator.generate_nb_val(clone_mean, clone_inv_dispersion, size=n_cells, rng_key=key)

            if simulate_neg_control:
                key, subkey = jax.random.split(key)
                x_neg = DextramerSimulator.generate_nb_val(mean_neg_ctrl, inv_dispersion_neg_ctrl,
                                                           size=n_cells, rng_key=key)
                d["x_neg"].extend(x_neg.tolist())

            d["x"].extend(x.tolist())
            d["is_binder"].extend([is_binder] * n_cells)
            d["clone_id"].extend([i] * n_cells)
            d["signal_to_noise"].extend([signal_to_noise] * n_cells)

        if p_dropout > 0:
            dropout = np.zeros(total_cells, dtype=int)
            is_binder_cell = np.array(d["is_binder"], dtype=bool)
            n_binder_cells = is_binder_cell.sum()

            dropout_trial = rng.binomial(1, p_dropout, size=(10000, n_binder_cells))

            err = np.abs(p_dropout - dropout_trial.mean(1))
            best_idx = err.argmin()
            dropout[is_binder_cell] = dropout_trial[best_idx]

            trunc_lower_bound = (0.001 - inv_dispersion_noise) / (inv_dispersion_noise / 3)
            clone_inv_dispersion = stats.truncnorm.rvs(trunc_lower_bound, np.inf, loc=inv_dispersion_noise,
                                                       scale=inv_dispersion_noise / 3, random_state=rng)
            x = np.array(d["x"])
            x[dropout.astype(bool)] = (
                DextramerSimulator.generate_nb_val(mean_noise, clone_inv_dispersion,
                                                   size=np.sum(dropout), rng_key=key)
            )
            d["x"] = x.tolist()
            d["dropout"] = dropout.tolist()
        else:
            d["dropout"] = [0] * total_cells

        mdat = DextramerSimulator.__generate_mdata(d, simulate_neg_control)
        # Best theoretical F1
        precision, recall, thresholds = precision_recall_curve(mdat['airr'].obs['is_binder'], mdat['gex'].X[:, 0])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]

        sim_params = {
            'mean_noise': mean_noise,
            'mean_signal': mean_signal,
            'mean_neg_ctrl': mean_neg_ctrl,
            'var_signal': var_signal,
            'inv_dispersion_noise': inv_dispersion_noise,
            'inv_dispersion_signal': inv_dispersion_signal,
            'inv_dispersion_neg_ctrl': inv_dispersion_neg_ctrl,
            'signal_to_noise': signal_to_noise,
            'signal_var_ratio': signal_var_ratio,
            'total_cells': total_cells,
            'n_clones': n_clones,
            'binder_fraction': binder_fraction,
            'p_dropout': p_dropout,
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
        hue = pd.Series(df["is_binder"]).map({0: "non-binder", 1: "binder"})
        # "extreme" here means an extreme count on the log scale, i.e. what the plot clips
        # for legibility -- unrelated to the simulated clonal `dropout`
        x_log = np.log(x + 1)  # Transform to log scale, roughly normal distributed
        zscore = (x_log - x_log.mean()) / x_log.std()
        x_no_extremes = x[zscore < Z_SCORE_CUTOFF]
        hue_no_extremes = hue[zscore < Z_SCORE_CUTOFF]
        extreme_thr = x_no_extremes.max()

        if "x_neg" in df:
            x_neg = df["x_neg"].values.reshape(-1, )
            x = np.concatenate((x, x_neg), axis=0)
            x_no_extremes = np.concatenate((x_no_extremes, x_neg), axis=0)
            hue = pd.concat([hue, pd.Series(["Neg control"]*len(x_neg))], axis=0)
            hue_no_extremes = pd.concat([hue_no_extremes, pd.Series(["Neg control"]*len(x_neg))], axis=0)

        n_cols = 3
        n_rows = 2
        fig = plt.figure(figsize=(3 * n_cols, 2.4 * n_rows))
        i = 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.axvline(extreme_thr, ls='--', c='r')
        sns.despine()
        plt.title('Linear (extreme count thr in red)')
        i += 1

        plt.subplot(n_rows, n_cols, i)
        ax = sns.histplot(x=x_no_extremes, discrete=True, stat='percent', element='step', hue=hue_no_extremes, hue_order=['non-binder', 'binder', 'Neg control'])
        sns.despine()
        plt.title(f'Linear, z-score < {Z_SCORE_CUTOFF}')
        i += 1

        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=False, title='Binding status')

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', binrange=(0, 100), element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.title('Linear x < 100')
        sns.despine()
        i += 1

        # Log scale
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.axvline(extreme_thr, ls='--', c='r')
        plt.title('Log-scale (extreme count thr in red)')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x_no_extremes, discrete=True, stat='percent', element='step', hue=hue_no_extremes, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title(f'Log-scale, z-score < {Z_SCORE_CUTOFF}')
        sns.despine()
        i += 1

        plt.subplot(n_rows, n_cols, i)
        sns.histplot(x=x, discrete=True, stat='percent', binrange=(0, 100), element='step', hue=hue, hue_order=['non-binder', 'binder', 'Neg control'], legend=False)
        plt.yscale('log')
        plt.title('Log-scale x < 100')
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

        adata.obs["signal_to_noise"] = d["signal_to_noise"]
        adata.obs.index = adata.obs.index.astype("int32")

        adata_tcr = ad.AnnData()
        adata_tcr.obs["is_binder"] = d["is_binder"]
        adata_tcr.obs["clone_id"] = d["clone_id"]
        adata_tcr.obs["dropout"] = d["dropout"]

        return md.MuData({"gex": adata, "airr": adata_tcr})

    @staticmethod
    def sample_var_from_mean(mean: Union[float, np.ndarray],
                             a: Optional[float] = None, b: Optional[float] = None,
                             resid_std: Optional[float] = None,
                             rng: Union[int, np.random.RandomState] = 42
                             ) -> Union[float, np.ndarray]:
        """
        Sample a realistic variance given a mean using the fitted power-law model:
            log(var) = a + b*log(mean) + Normal(0, resid_std^2)

        Args:
            mean : float or np.ndarray
                Mean(s) at which to sample the variance. Must be > 0; broadcasting allowed.
            a : float, optional
                Proportionality constant (exp(intercept) from log–log OLS).
            b : float, optional
                Scaling exponent (slope from log–log OLS).
            resid_std : float, optional
                Residual standard deviation on the *log-variance* scale (σ from OLS residuals).
                `a`, `b` and `resid_std` default to `default_params()`'s
                `mean_var_powerlaw`.
            rng : int | np.random.RandomState, default 42
                Source of randomness. If int, used as the seed. If None, uses SciPy/Numpy default RNG.
        Returns:
            float or np.ndarray
                A sample of variance values with the same broadcasted shape as `mean`.
        """
        fitted_a, fitted_b, fitted_resid_std = DextramerSimulator.default_params()["mean_var_powerlaw"]
        a = fitted_a if a is None else a
        b = fitted_b if b is None else b
        resid_std = fitted_resid_std if resid_std is None else resid_std

        if isinstance(rng, int):
            rng = np.random.RandomState(seed=rng)
        if isinstance(mean, np.ndarray):
            size = mean.shape
        else:
            size = None
        log_var = np.log(a) + b*np.log(mean) + stats.norm(0, resid_std).rvs(size=size, random_state=rng)
        var = np.exp(log_var)

        return var

    @staticmethod
    def generate_nb_val(mu, alpha, size=1, rng_key=42):
        """Generate negative binomial samples

        Args:
            mu: the mean parameter (must be positive)
            alpha: the inverse dispersion parameter (must be positive)
            size: the number of iid draws
            rng_key: int or jax.random.PRNGKey as random seed
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)
        return npd.NegativeBinomial2(mu, alpha).sample(rng_key, sample_shape=(size,))
