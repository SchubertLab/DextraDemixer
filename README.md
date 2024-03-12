# DextraMixer Package (Placeholder Name)

This module implements several mixture models to infer pMHC dextramere specificity from single-cell immune profiling data
with increasing usage of information

**Given**:

A read count matrix **$X_{ij}\in \mathbb{N}$** approximating the avidity of $i\in N$ T cell for the $j\in M$ epitope. The $N$ T cells can be grouped based on their T-cell receptor sequence into $C$ cluster.

**Assumption**:

1) Each read counts $X_{.j}$ of an epitope is iid.
2) We assume that $X_{.j}$ can be represented as a mixture of two Negative Binomial distributions. Each clonal group $c \in C$ belongs to either the clone-specific **Binding** or the **Non-Binding** component. The **Non-Binding** component represents unspecific epitope binding and assay noise.
3) It is assumed that unspecific binding T cells and non-binding T cells exhibit lower read counts compared to specifically binding T cell after appropriate normalization.
4) T cells of a clonal group $c\in C$ are drawn from the same distribution.

# Help needed!

I am looking for volunteers who are interested in developing and testing this package with
me. TODOS will be managed as issues.
