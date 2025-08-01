from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import os

import mudata as md
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import jax.lax
import jax

from dextrademixer.model import ApMHCDeconvolution

if TYPE_CHECKING:
    from jax._src.typing import Array


class BEAMT(ApMHCDeconvolution):
    """
    This class implements the BEAM-T algorithm used by 10x Genomics.
    It requires a negative control besides the pMHC-dextramer and calculates an antigen-specificity score using
    a Beta distribution parameterized by the UMI counts of the pMHC and negative control.

    p = (1-beta.cdf(quantile, pMHC-UMI+1, neg_ctrl-UMI+3))
    """
    __name = "BEAMT"
    __version = "0.0.1"

    def __init__(self):
        super().__init__()
        self.params = None
        self.p = None
        self.data = None

    def preprocess_model_data(self, mdata: md.MuData, pmhc_key: str, gex_key: str = "gex", neg_ctrl_key: str = None,
                              ir_key: str = "airr", ir_clone_key: str = None, ir_cov_key: str = None, **kwargs):
        if neg_ctrl_key is None:
            raise ValueError(f"{self.__name} requires a negative control. Please specify a `neg_ctrl_key`.")

        gex = mdata.mod[gex_key]
        N = gex.shape[0]

        x = gex[:, pmhc_key].X.toarray().reshape((N,))
        x_neg = gex[:, neg_ctrl_key].X.toarray().reshape((N,))

        self._check_parameters(x, x_neg, None, None)

        self.data = {"x": x, "x_neg": x_neg}

        self.params = {"alpha": x+1, "beta": x_neg+3}

    def fit(self, percentile: float = 0.925):
        """
        Args:
            percentile: the percentile which is used to classify pMHC dextramers as binder
        """
        if self.params is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.p = 1 - jax.scipy.stats.beta.cdf(percentile, self.params["alpha"], self.params["beta"])

    def predict_posterior_class(self, threshold: float = None, target_fdr: float = None) -> Tuple[Array, Array]:
        """
        Returns the binder assignments based on the inferred posterior class probabilities.
        Assignment can be either be done by providing a threshold or target fdr value if FDR control is wanted.
        If neither threshold nor target_fdr is provided the max posterior class probability will be used.

        Args:
             threshold: (Optional) a threshold in [0,1] determining binder based on inferred posterior class
                        probabilities
            target_fdr: (Optional) the FDR threshold to control False discovery rate based on the posterior
                        class probability
        Returns:
            A tuple (p, assignment) of arrays with p being the posterior probability of binding and assignment the
            class assignment decision
        """
        if self.p is None:
            raise RuntimeError("Model has not been fit yet. Please call first `fit`.")

        # posterior probability of belonging to the binding class
        assignment = self._predict_posterior_class(self.p, threshold, target_fdr)
        return self.p.__array__(), assignment.__array__()

    def plot_results(self, assignment, p_pred, y_true=None, seed=42, config=''):

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
