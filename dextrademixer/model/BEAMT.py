from __future__ import annotations

import os

import numpy as np

import mudata as md
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import jax.lax
import jax

from dextrademixer.model import ApMHCDeconvolution

class BEAMT(ApMHCDeconvolution):
    """
    Assign pMHC binding with the 10x Genomics BEAM-T algorithm.

    Attributes
    ----------
    params : dict or None
        Beta-distribution parameters derived from target and control counts.
    p : numpy.ndarray or None
        Cell-level antigen-specificity probabilities after fitting.
    data : dict or None
        Preprocessed target and negative-control counts.

    Notes
    -----
    BEAM-T requires a negative-control feature. It parameterizes a beta
    distribution with target and negative-control UMI counts and calculates the
    upper-tail probability at a selected percentile.
    """
    __name = "BEAMT"
    __version = "0.0.1"

    def __init__(self):
        super().__init__()
        self.params = None
        self.p = None
        self.data = None

    def preprocess_model_data(self, mdata: md.MuData, pmhc_key: str, gex_key: str = "gex", neg_ctrl_key: str = None,
                              ir_key: str = "airr", ir_clone_key: str = None, ir_cov_key: str = None,
                              **kwargs) -> None:
        """
        Extract target and negative-control counts from a MuData object.

        Parameters
        ----------
        mdata : mudata.MuData
            Object containing the pMHC count modality.
        pmhc_key : str
            Target pMHC feature name.
        gex_key : str, default="gex"
            Key of the modality containing pMHC counts.
        neg_ctrl_key : str
            Negative-control feature name.
        ir_key : str, default="airr"
            Unused immune-receptor modality key retained for interface
            compatibility.
        ir_clone_key : str, optional
            Unused clonotype column retained for interface compatibility.
        ir_cov_key : str, optional
            Unused covariance key retained for interface compatibility.
        **kwargs : object
            Additional unused values accepted for interface compatibility.

        Raises
        ------
        ValueError
            If ``neg_ctrl_key`` is missing or extracted arrays are inconsistent.
        """
        if neg_ctrl_key is None:
            raise ValueError(f"{self.__name} requires a negative control. Please specify a `neg_ctrl_key`.")

        gex = mdata.mod[gex_key]
        N = gex.shape[0]

        x = gex[:, pmhc_key].X.toarray().reshape((N,))
        x_neg = gex[:, neg_ctrl_key].X.toarray().reshape((N,))

        self._check_parameters(x, x_neg, None)

        self.data = {"x": x, "x_neg": x_neg}

        self.params = {"alpha": x+1, "beta": x_neg+3}

    def fit(self, percentile: float = 0.925) -> None:
        """
        Calculate cell-level antigen-specificity probabilities.

        Parameters
        ----------
        percentile : float, default=0.925
            Beta-distribution quantile at which the upper-tail probability is
            evaluated.

        Raises
        ------
        Exception
            If :meth:`preprocess_model_data` has not been called.
        """
        if self.params is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.p = 1 - jax.scipy.stats.beta.cdf(percentile, self.params["alpha"], self.params["beta"])

    def predict_posterior_class(self, threshold: float = None,
                                target_fdr: float = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return binding probabilities and binary assignments.

        Parameters
        ----------
        threshold : float, optional
            Probability threshold in ``[0, 1]``. Defaults to 0.5 when neither
            assignment option is specified.
        target_fdr : float, optional
            Target Bayesian false discovery rate in ``[0, 1]``. Mutually
            exclusive with ``threshold``.

        Returns
        -------
        p : numpy.ndarray
            Cell-level antigen-specificity probabilities.
        assignment : numpy.ndarray
            Binary binding assignments.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.p is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit`.")

        # posterior probability of belonging to the binding class
        assignment = self._predict_posterior_class(self.p, threshold, target_fdr)
        return self.p.__array__(), assignment.__array__()

    def plot_results(self, assignment, p_pred, y_true=None, seed=42, config='') -> None:
        """
        Plot target counts and predicted and true assignments.

        Parameters
        ----------
        assignment : array-like
            Predicted binary assignments.
        p_pred : array-like
            Cell-level antigen-specificity probabilities.
        y_true : array-like, optional
            True labels used for comparison and F1 calculation.
        seed : int, default=42
            Reserved random seed retained for API compatibility.
        config : str, default=""
            Label used in the figure title and output filename.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.

        Notes
        -----
        The figure is saved under ``figs/{config}.png`` and then displayed.
        """

        if self.p is None:
            raise RuntimeError("Model has not been fit yet. Please call `fit` or `fit_svi` first.")

        plt.figure(figsize=(6, 12))

        # Plot data colored in predicted class assignment
        plt.subplot(3, 2, 1)
        sns.histplot(x=self.data["x"], hue=assignment,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.title("Predicted class assignment")

        plt.subplot(3, 2, 3)
        sns.histplot(x=self.data["x"], hue=assignment,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.yscale("log")
        plt.title("Predicted class assignment log-scale")

        plt.subplot(3, 2, 5)
        sns.scatterplot(x=self.data["x"], y=p_pred, hue=assignment,
                        markers={0: ".", 1: "X"})
        sns.despine()
        plt.ylabel("Posterior probability")
        plt.title("Predicted probability and label of UMI count")

        # Plot data colored in true class assignment
        plt.subplot(3, 2, 2)
        sns.histplot(x=self.data["x"], hue=y_true,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.title("True class assignment")

        plt.subplot(3, 2, 4)
        sns.histplot(x=self.data["x"], hue=y_true,
                     discrete=True, element="step", alpha=0.7)
        sns.despine()
        plt.yscale("log")
        plt.title("True class assignment log-scale")

        plt.subplot(3, 2, 6)
        sns.scatterplot(x=self.data["x"], y=p_pred, hue=y_true)
        sns.despine()
        plt.ylabel("Posterior probability")
        plt.title("Predicted probability of UMI count with true label")

        # Save plot
        try:
            f1 = f1_score(assignment, y_true)
        except:
            # if y_true is None or str
            f1 = -1
        plt.suptitle(config.replace("_", " ").replace("ncell", "\nncell") + f"\nF1-score {f1:.3f}",)
        os.makedirs("figs", exist_ok=True)
        plt.savefig(f"figs/{config}.png")
        plt.show()
        plt.close()
