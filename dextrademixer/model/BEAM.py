from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import os

import numpy as np

import mudata as md
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import jax.lax
import jax

from dextrademixer.model.ApMHCDeconvolution import ApMHCDeconvolution, Data

if TYPE_CHECKING:
    from jax._src.typing import Array


class BEAM(ApMHCDeconvolution):
    """
    This class implements the BEAM-T algorithm used by 10x Genomics.
    It requires a negative control besides the pMHC-dextramer and calculates an antigen-specificity score using
    a Beta distribution parameterized by the UMI counts of the pMHC and negative control.

    p = (1-beta.cdf(quantile, pMHC-UMI+1, neg_ctrl-UMI+3))
    """
    __name = "BEAM"
    __version = "0.0.1"

    def __init__(self, percentile: float = 0.925):
        """
        Args:
            percentile: the percentile which is used to classify pMHC dextramers as binder
        """
        super().__init__()
        self.percentile = percentile
        self.params = None
        self.p = None
        self.data = None

    def preprocess_model_data(self, data: Data, pmhc_key: str, pmhc_modality_key: str = "gex", neg_ctrl_key: str = None,
                              ir_modality_key: str = "airr", ir_clone_key: str = None, **kwargs):
        """
        Pulls the pMHC and negative control counts out of `data`.

        Args:
            data: the pMHC counts, as a MuData, an AnnData or a cells x features DataFrame.
                  See `as_counts` for where the counts are read from in each case;
                  `pmhc_modality_key`/`ir_modality_key` are only used for MuData.
            pmhc_key: the pMHC count column to deconvolve
            pmhc_modality_key: the MuData modality holding the counts
            neg_ctrl_key: the negative control count column, required by BEAM
            ir_modality_key: (Optional) unused, the MuData AIRR module key
            ir_clone_key: (Optional) unused, accepted for interface parity

        Raises:
            ValueError: if `neg_ctrl_key` is None.
        """
        if neg_ctrl_key is None:
            raise ValueError(f"{self.__name} requires a negative control. Please specify a `neg_ctrl_key`.")

        counts, _ = self.as_counts(data, pmhc_modality_key, ir_modality_key)

        x = counts[pmhc_key].to_numpy()
        x_neg = counts[neg_ctrl_key].to_numpy()

        self._check_parameters(x, x_neg, None)

        self.data = {"x": x, "x_neg": x_neg}

        self.params = {"alpha": x+1, "beta": x_neg+3}

    def fit_scores(self):
        """
        Scores the data prepared by `preprocess_model_data` and stores the Beta posterior.

        Raises:
            Exception: if `preprocess_model_data` has not been called yet.
        """
        if self.params is None:
            raise Exception("Model is not initialized. Please call `preprocess_model_data` first.")

        self.p = 1 - jax.scipy.stats.beta.cdf(self.percentile, self.params["alpha"], self.params["beta"])

    def fit(self, data: Data, *, pmhc_key: str = None, pmhc_modality_key: str = "gex",
            neg_ctrl_key: str = None, ir_modality_key: str = "airr",
            ir_clone_key: str = None) -> "BEAM":
        """
        Extracts the model data from `data` and fits it, i.e. `preprocess_model_data` followed by
        `fit_scores`.

        Args:
            data: the pMHC counts, as a MuData, an AnnData or a cells x features DataFrame
            pmhc_key: the pMHC count column to deconvolve
            pmhc_modality_key: the MuData modality holding the counts
            neg_ctrl_key: the negative control count column, required by BEAM
            ir_modality_key: (Optional) unused, the MuData AIRR module key
            ir_clone_key: (Optional) unused, the `obs` column that holds clonotype ids

        Returns:
            self, so that `predict` can be chained onto the call
        """
        self.preprocess_model_data(data, pmhc_key=pmhc_key, pmhc_modality_key=pmhc_modality_key,
                                   neg_ctrl_key=neg_ctrl_key, ir_modality_key=ir_modality_key,
                                   ir_clone_key=ir_clone_key)
        self.fit_scores()
        return self

    def predict_posterior_class(self, threshold: float = None, target_fdr: float = None) -> Tuple[np.array, np.array]:
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
