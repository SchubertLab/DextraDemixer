# DextraDemixer

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.23.733339-b31b1b)](https://doi.org/10.64898/2026.06.23.733339)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/SchubertLab/DextraDemixer/blob/main/LICENSE)

**Probabilistic identification of antigen-specific T cells from single-cell pMHC multimer experiments.**

DextraDemixer models pMHC multimer UMI counts as a mixture of specific and nonspecific binding components. It returns a posterior binding probability and assignment for each cell, helping separate antigen-specific signal from background binding and assay noise.

The method is described in [*DextraDemixer enables accurate identification of antigen-specific T cells from pMHC multimer experiments*](https://doi.org/10.64898/2026.06.23.733339).

> [!NOTE]
> DextraDemixer is research software under active development. Interfaces may change while the package matures.

## Highlights

- Infers cell-level posterior probabilities of antigen specificity from pMHC UMI counts.
- Supports optional negative-control multimers.
- Incorporates clonotype information through clone-level probability aggregation.
- Provides fixed-threshold and Bayesian false discovery rate (FDR) assignments.
- Works directly with [`MuData`](https://mudata.readthedocs.io/), [`AnnData`](https://anndata.readthedocs.io/), and [`Pandas DataFrames`](https://pandas.pydata.org/) objects and uses [JAX](https://jax.readthedocs.io/) and [NumPyro](https://num.pyro.ai/) for inference.

## Installation

```bash
pip install dextrademixer
```

### Alternative: Conda environment

For a conda environment.

```bash
git clone https://github.com/SchubertLab/DextraDemixer.git
cd DextraDemixer
conda env create -f environment.yaml
conda activate dextrademixer
pip install -e .  # or pip install dextrademixer
```

The final command installs the local package in editable mode. Runtime dependency constraints are aligned between `pyproject.toml` and `environment.yaml`. To reproduce the numbers from the manuscript, use `environment_reproducible.yaml` instead: it pins the exact versions, and results can shift slightly with other JAX/NumPyro releases.

## Quick start

### Single pMHC

The example below fits one pMHC feature and classifies cells using a posterior-probability threshold of 0.5.

```python
import muon as mu

from dextrademixer import DextraDemixer

mdata = mu.read("data/example_data.h5mu")

model = DextraDemixer().fit(mdata, pmhc_key="pmhc1", pmhc_modality_key="gex")
p_binder, is_binder = model.predict(threshold=0.5)

mdata.mod["gex"].obs["dextrademixer_probability"] = p_binder
mdata.mod["gex"].obs["dextrademixer_assignment"] = is_binder
```

To include a negative-control multimer, pass `neg_ctrl_key`. To aggregate probabilities within clonotypes, pass the clonotype column as `ir_clone_key` and set `clonotype_median_p=True` when predicting:

```python
model = DextraDemixer().fit(
    mdata,
    pmhc_key="pmhc1",
    pmhc_modality_key="gex",
    neg_ctrl_key="neg_control",
    ir_clone_key="clone_id",
)

p_pred, assignment = model.predict(threshold=0.5, clonotype_median_p=True)
```

Use either `threshold` for a fixed decision boundary or `target_fdr` for FDR-controlled assignments; do not specify both.

### Multiple pMHCs

`DextraDemixerMulti` fits one independent `DextraDemixer` per pMHC and returns cells x pMHC tables instead of vectors. Each fit stays reachable as `model.demixers[pmhc_key]`.

```python
from dextrademixer import DextraDemixerMulti

mdata = mu.read("data/example_multi_pmhc.h5mu")

model = DextraDemixerMulti().fit(
    mdata,
    pmhc_keys=["pmhc1", "pmhc2", "pmhc3"],
    pmhc_modality_key="gex",
    neg_ctrl_key="neg_control",
)

p_pred, assignment = model.predict(threshold=0.5, clonotype_median_p=True, max_prob=True)
```

Because the pMHCs are fitted independently, a cell can pass the threshold for several. `max_prob=True` keeps only the pMHC with the highest probability per cell, combined with `clonotype_median_p=True` the same choice is made per clonotype.

## Input data

DextraDemixer needs the pMHC UMI counts and, optionally, a cell-level clonotype identifier. It reads them from any of:

- A [`MuData`](https://mudata.readthedocs.io/) object: counts in the `.X` of the feature modality (`pmhc_modality_key`, `gex` by default), clonotypes in the `.obs` of the AIRR modality (`ir_modality_key`, `airr` by default).
- An [`AnnData`](https://anndata.readthedocs.io/) object: counts in `.X`, clonotypes in `.obs`; `pmhc_modality_key` and `ir_modality_key` are then unused.
- A cells x features [`DataFrame`](https://pandas.pydata.org/): counts and annotation in the same table, so every key is a column name.

`pmhc_key` (or `pmhc_keys`) and the optional `neg_ctrl_key` name the count columns, `ir_clone_key` the clonotype column, whose ids may be integers or strings. The bundled example dataset ships in all three formats: `data/example_data.h5mu`, `.h5ad` and `.csv`. `data/example_multi_pmhc.h5mu` is a simulated three-pMHC panel.

See [`Tutorial.ipynb`](https://github.com/SchubertLab/DextraDemixer/blob/main/Tutorial.ipynb) for a complete, reproducible workflow using the bundled example dataset, including configuration, model fitting, evaluation, and visualization, and [`Tutorial DextraDemixerMulti.ipynb`](https://github.com/SchubertLab/DextraDemixer/blob/main/Tutorial%20DextraDemixerMulti.ipynb) for the same with multiple pMHCs:

```bash
jupyter lab Tutorial.ipynb
```

## Reproducing the manuscript

The analyses and figure-generation code used in the manuscript are maintained in the [DextraDemixer reproducibility repository](https://github.com/SchubertLab/DextraDemixer_reproducibility).

## Contributing

Bug reports, feature requests, and pull requests are welcome through the [GitHub issue tracker](https://github.com/SchubertLab/DextraDemixer/issues). When reporting a problem, please include a minimal reproducible example and details of your operating system and Conda environment.

## Citation

If DextraDemixer supports your research, please cite:

```bibtex
@article{An2026DextraDemixer,
  author       = {An, Yang and Drost, Felix and Bonafonte-Pard{\`a}s, Irene and Grotz, Myriam and Schober, Kilian and Schubert, Benjamin},
  title        = {DextraDemixer enables accurate identification of antigen-specific T cells from pMHC multimer experiments},
  journal      = {bioRxiv},
  year         = {2026},
  publisher    = {Cold Spring Harbor Laboratory},
  doi          = {10.64898/2026.06.23.733339},
  url          = {https://doi.org/10.64898/2026.06.23.733339},
  elocation-id = {2026.06.23.733339}
}
```

## License

DextraDemixer is distributed under the [BSD 3-Clause License](https://github.com/SchubertLab/DextraDemixer/blob/main/LICENSE).
