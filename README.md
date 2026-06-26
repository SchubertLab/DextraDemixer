# DextraDemixer

DextraDemixer is a Python package for identifying antigen-specific T cells from pMHC multimer experiments.

The package implements the mixture model described in **DextraDemixer enables accurate identification of antigen-specific T cells from pMHC multimer experiments** [link](https://www.biorxiv.org/content/10.64898/2026.06.23.733339v1)

DextraDemixer models pMHC multimer UMI counts to distinguish antigen-specific binders from nonspecific binders, enabling more accurate identification of T cells recognizing specific peptide–MHC targets.

DextraDemixer is under active development. We are continuously improving the usability, documentation, and functionality of the package. Feedback and contributions are welcome.

## Installation and Tutorial

Please execute the following to install DextraDemixer:

```bash
git clone git@github.com:SchubertLab/DextraDemixer.git
cd DextraDemixer
conda env create -f environment.yaml
```

A tutorial can be found in `Tutorial.ipynb`.

## Citation

If you found this tool helpful for your research, please cite:

```bibtex
@article {An2026DextraDemixer,
author = {An, Yang and Drost, Felix and Bonafonte-Pard{\`a}s, Irene and Grotz, Myriam and Schober, Kilian and Schubert, Benjamin},
title = {DextraDemixer enables accurate identification of antigen-specific T cells from pMHC multimer experiments},
elocation-id = {2026.06.23.733339},
year = {2026},
doi = {10.64898/2026.06.23.733339},
publisher = {Cold Spring Harbor Laboratory},
URL = {https://www.biorxiv.org/content/early/2026/06/25/2026.06.23.733339},
journal = {bioRxiv}
}
```
