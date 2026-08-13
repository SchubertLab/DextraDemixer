# DextraDemixer

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.23.733339-b31b1b)](https://doi.org/10.64898/2026.06.23.733339)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

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
- Works directly with [`MuData`](https://mudata.readthedocs.io/) objects and uses [JAX](https://jax.readthedocs.io/) and [NumPyro](https://num.pyro.ai/) for inference.

## Installation

The recommended setup uses the curated Conda environment included in this repository. Creating it can take approximately 10 minutes.

```bash
git clone https://github.com/SchubertLab/DextraDemixer.git
cd DextraDemixer
conda env create -f environment.yaml
conda activate dextrademixer
python -m pip install -e .
```

The final command installs the local package in editable mode. Runtime dependency versions are aligned between `pyproject.toml` and `environment.yaml`.

## Quick start

The example below fits one pMHC feature and classifies cells using a posterior-probability threshold of 0.5.

```python
import muon as mu

from dextrademixer import DextraDemixer

mdata = mu.read("data/example_data.h5mu")

model = DextraDemixer()
model.preprocess_model_data(
    mdata,
    pmhc_key="pmhc1",
    gex_key="gex",
)
model.fit_svi(nof_inits=10, rng_key=42)

p_binder, is_binder = model.predict_posterior_class(threshold=0.5)
mdata.mod["gex"].obs["dextrademixer_probability"] = p_binder
mdata.mod["gex"].obs["dextrademixer_assignment"] = is_binder
```

To include a negative-control multimer, pass `neg_ctrl_key` to `preprocess_model_data`. To aggregate probabilities within clonotypes, pass `clonotype_median_p=True` and a cell-aligned `clone_id` array to `predict_posterior_class`. For example:

```python
model.preprocess_model_data(
    mdata,
    pmhc_key="pmhc1",
    gex_key="gex",
    neg_ctrl_key="neg_control",
)
model.fit_svi(nof_inits=10, rng_key=42)

p_binder, is_binder = model.predict_posterior_class(
    threshold=0.5,
    clonotype_median_p=True,
    clone_id=mdata.mod["airr"].obs["clone_id"].to_numpy(),
)
```

Use either `threshold` for a fixed decision boundary or `target_fdr` for FDR-controlled assignments; do not specify both.

## Input data

DextraDemixer expects a `MuData` object with cell-aligned modalities:

- A feature-count modality (called `gex` by default) containing the pMHC UMI counts in `.X`; `pmhc_key` and the optional `neg_ctrl_key` refer to its feature names.
- An AIRR modality (called `airr` by default). Its `.obs` can provide a cell-level clonotype identifier for clone-aware prediction.

See [`Tutorial.ipynb`](Tutorial.ipynb) for a complete, reproducible workflow using the bundled example dataset, including configuration, model fitting, evaluation, and visualization:

```bash
jupyter lab Tutorial.ipynb
```

## Building the API documentation

Install the documentation dependencies together with the package, then build the
Sphinx site with warnings treated as errors:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the generated API reference.

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

DextraDemixer is distributed under the [BSD 3-Clause License](LICENSE).
