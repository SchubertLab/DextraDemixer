rom __future__ import annotations

#from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
#import scirpy as ir
#from adjustText import adjust_text
from anndata import AnnData
from jax import random
from jax.config import config
#from mudata import MuData
from numpyro.infer import HMC, MCMC, NUTS, initialization


if TYPE_CHECKING:
    import numpyro as npy
    import numpyro.distributions as npd
    from jax._src.prng import PRNGKeyArray
    from jax._src.typing import Array

config.update("jax_enable_x64", True)

class DexaMix:
    """
    This class implements several mixture models to infere pMHC dextramere speceficity from single cell immuneprofiling data
    with increasing usage of information 

    **Given**: 
        
    A read count matrix ***$K_{ij}\in \mathbb{N}$*** approximating the avidity of $i\in N$ T cell for the $j\in M$ epitope. The $N$ T cells can be grouped based on their T-cell receptor sequence into $C$ cluster.
        
    **Assumption**:
        
    1) Each read counts $K_{.j}$ of an epitope is iid.
    2) We assume that $K_{.j}$ can be represented as a mixture of two Negative Binomial distributions. Each clonal group $c \in C$ belongs to either the clone-specific **Binding** or the **Non-Binding** component. The **Non-Binding** component represents unspecific epitope binding and assay noise.
    3) It is assumed that unspecific binding T cells and non-binding T cells exhibit lower read counts compared to specifically binding T cell after appropriat normalization.
    4) T cells of a cloncal group $c\in C$ are drawn from the same distribution.

    """

    def __simple_mixture(  # type: ignore
        self,
        counts: np.ndarray,
        size_factor: np.ndarray,
        sample_adata: AnnData,):
        """
        implements a simple NegBinom Mixture model with one component representing noise and one singal

        Args:
            counts: Count data matrix
            size_factor: Normalization size factor data array
            sample_adata: Anndata object with Dextramer read counts per cell as sample_adata.X
        Returns:
            predictions (see numpyro documentation for details on models)
        """
        N,M = sample_adata.X.shape
        K = 2

        #plates
        dexa_axis = npy.plate("dexa", M, dim=-2)
        sample_axis = npy.plate("sample", N, dim=-1)
        cluster_axis = npy.plate("K", K, dim=-1)
        
        #set hyperprior
        mu_q = npy.sample("mu_q", npd.Normal(0,10))
        sigma_q = npy.sample("sigma_q", npd.HalfCauchy(5))
        w = npy.sample("w", npd.Beta(1,1))
        z = npd.Categorial(probs=jnp.array([1.0 - w, w]))
        
        # set cluster prior
        with dexa_axis, cluster_axis:
            q_raw = npy.sample("q_raw", 
                               npd.TransformedDistribution(
                                   npd.LogNormal(mu_q, sigma_q), npd.transforms.OrderedTransform()
                               ))
            alpha = npy.sample("alpha", npd.HalfNormal(1)) 
            pyro.factor("power_law", 1/jnp.sqrt(alpha)) #according to stan standard prior

        with dexa_axis, sample_axis:

            nois_dist = 
            mixture = npd.MixtureSameFamily(z, [fg_dist, bg_dist])

            
            yhat = 

            # Until here, where we can track the membership probability of each sample
            log_probs = mixture.component_log_probs(y_)
            numpyro.deterministic(
                "p", log_probs - jnp.nn.logsumexp(log_probs, axis=-1, keepdims=True)
            )
        
            


    def __clonotype_model(self):
        """
        implements a hierarchical mixture model on the level of clonotypes

        **Model**:
        
        The generative model is wlog. specified for data of an individual epitope $j$.
        
        $$\begin{align}
        \text{\# Hyperprior}\\
        \mu_q &\sim \text{N}(0,1)\\
        \sigma_q &\sim \text{HC}(5)\\ 
        q_{\text{raw}}^b  &\sim \text{N}(\mu_q, \sigma_q) &\forall b \in \{0,1\}\\ 
        \sigma_q^b  &\sim \text{HC}(5) &\forall b \in \{0,1\}\\
        p_c &\sim \text{Beta}(0.5,0.5) &\forall c \in C \text{ \# Extention with Logit-N T-cell similarity}\\
        q_c &\sim (1-p_c)\text{N}(q_{\text{raw}}^0, \sigma_q^0)+p_c\text{N}(q_{\text{raw}}^1, \sigma_q^1) &\forall c \in C \text{ \# Extention with MvN T-cell similarity}\\
        \text{\# Stan standard prior for NB shape parameter}&\\
        \frac{1}{\sqrt{\alpha_c}} &\sim\text{half-N}(0,1) &\forall c \in C \\
        \text{\# Likelihood}&\\
        y_{C(i)} &\sim \text{NegBinom}(s_i e^{q_{C(i)}}, \alpha_{C(i)}) &\forall i \in N\\
        \text{s.t.  }  q_{\text{raw}}^0 &< q_{\text{raw}}^1\\
        \end{align}$$
        """
        pass
        with pm.Model(coords={"obs":df.index, "clone":cluster_name, "cluster":np.arange(k)}) as basic_model:
            y = pm.pytensorf.intX(pm.ConstantData("y", df[['clone','avidity']].values))
            
            #hyper prior
            mu_q = pm.Normal("mu_q",0,10)
            sigma_q = pm.HalfCauchy("sigma_q",5)
        
            #mean prior
            qs_raw = pm.Normal("q_raw", mu=mu_q, sigma=sigma_q, 
                               initval=[-1, 1],
                               transform=pm.distributions.transforms.univariate_ordered, dims="cluster") 
            sigma_q_c = pm.HalfCauchy("sigma_q_c",5, dims="cluster")    
        
            #shape prior
            alpha_c = pm.HalfNormal("alpha_c",1, dims="clone")
            pm.Potential("power_alpha_c", 1/pm.math.sqrt(alpha_c))
            
            p = pm.Beta("w", 1,1, dims="clone")
            
            #likelihood
            for i, clone in df.groupby("clone"):
                
                w =[1-p[i],p[i]]
                q = pm.NormalMixture("mix_q_%i"%i, 
                                w=w, 
                                mu=qs_raw,
                                sigma=sigma_q)
                yhat = pm.NegativeBinomial("yhat_%i"%i,mu=size_factor*pm.math.exp(q), alpha=alpha_c[i], observed=clone['avidity'].values)
                
    def __sequence_kernel_model(self):
        """
        implements model that takes sequence similarity into account following assumption: 
        
        5) Clones with similar TCR sequences have similar binding avidity

        """
        pass

    def infere_noise_prior(self):
        """
        uses negative controll dextramers to infere noise level in experiment 
        and uses this as informative prior in models
        """
        pass
    
    
    def load(
        self,
        ...
    )
    """
    Prepares data input and chosses specific DexaMix model according to presented additional information
    
    """
    pass

    def run_nuts(
        self,
        data: AnnData | MuData,
        modality_key: str = "DexaMix",
        num_samples: int = 10000,
        num_warmup: int = 1000,
        rng_key: int = 0,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Run No-U-turn sampling (Hoffman and Gelman, 2014), an efficient version of Hamiltonian Monte Carlo sampling to infer optimal model parameters.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            num_samples: Number of sampled values after burn-in. Defaults to 10000.
            num_warmup: Number of burn-in (warmup) samples. Defaults to 1000.
            rng_key: The rng state used. Defaults to 0.
            copy: Return a copy instead of writing to adata. Defaults to False.

        Returns:
            Calls `self.__run_mcmc`
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if copy:
            sample_adata = sample_adata.copy()

