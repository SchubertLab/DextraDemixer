"""Sphinx configuration for the DextraDemixer documentation."""

project = "DextraDemixer"
copyright = "2026, DextraDemixer contributors"
author = "DextraDemixer contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_preserve_defaults = True
autodoc_typehints = "description"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_validation_checks = {"all", "EX01", "SA01", "ES01"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"

