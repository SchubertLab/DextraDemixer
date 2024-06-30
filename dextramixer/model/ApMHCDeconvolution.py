from __future__ import annotations

import abc

from typing import TYPE_CHECKING, Tuple

import mudata as md

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax._src.typing import Array


class ApMHCDeconvolution:

    @abc.abstractmethod
    def preprocess_model_data(self,
                              mdata: md.MuData,
                              pmhc_key: str,
                              gex_key: str = "gex",
                              neg_ctrl_key: str = None,
                              ir_key: str = "airr",
                              ir_clone_key: str = None,
                              ir_cov_key: str = None,
                              **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_posterior_class(self,
                                threshold: float = None,
                                target_fdr: float = None
                                ) -> Tuple[Array, Array]:
        pass

    @staticmethod
    def _predict_posterior_class(p: Array,
                                 threshold: float = None,
                                 target_fdr: float = None
                                 ) -> Array:

        if threshold is not None and target_fdr is not None:
            raise ValueError("Please specify either a manual `threshold` or a `target_fdr` but not both.")

        if threshold is not None and not (0 <= threshold <= 1):
            raise ValueError(f"`threshold`must be in [0,1] but was {threshold}")

        if target_fdr is not None and not (0 <= target_fdr <= 1):
            raise ValueError(f"`target_fdr`must be in [0,1] but was {target_fdr}")

        if threshold is None and target_fdr is None:
            threshold = 0.5

        # posterior probability of belonging to the binding class
        if target_fdr is not None:
            # Direct posterior probability approach cf. Newton et al.(2004)
            def opt_thresh(p_, alpha):

                incs = p_.copy()
                incs = incs[::-1].sort()

                for c in jnp.unique(incs):
                    fdr = jnp.mean(1 - incs[incs >= c])
                    if fdr < alpha:
                        return c, fdr
                return 1., 0

            threshold, fdr_ = opt_thresh(p, target_fdr)

        assignment = (p >= threshold).astype("int32")
        return assignment


    @staticmethod
    def _check_parameters(x, neg_x, c, sigma):
        """
        checks consistency of input data before initializing the model
        """
        N = x.shape[0]

        if c is not None:
            if c.shape[0] != N:
                raise ValueError(f"`c` and count data `x` require the same size but got {c.shape[0]} and {N}")

        if neg_x is not None:
            N_neg = neg_x.shape[0]

            if N_neg != N:
                raise ValueError(f"x_neg must have the same size than x but got {N_neg} vs {N}.")

        if sigma is not None:
            if c is None:
                raise ValueError("If `sigma` is given, clonality vector `c` must be given as well")
            else:
                C_nof = len(jnp.unique(c))
                if sigma.shape[0] != C_nof:
                    raise ValueError(f"Sigma must have shape ({C_nof},{C_nof}) and defined over clonotypes but has"
                                     + f"{sigma.shape}")

