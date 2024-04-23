# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../"))  # noqa: PTH100
# sys.path.insert(0, os.path.abspath("../tutorials/"))
autodoc_mock_imports = [
    "energyflow",
    "awkward",
    "coffea",
    "tqdm",
    "scipy",
    "torch_geometric",
    "torch",
    "cvxpy",
    "qpth",
    "numba",
]

# -- Project information -----------------------------------------------------

project = "JetNet"
copyright = "2021, Raghav Kansal"
author = "Raghav Kansal"

# The full version, including alpha/beta/rc tags
release = "0.2.0a"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "m2r2",
    "nbsphinx",
    "sphinx_rtd_theme",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

autodoc_type_aliases = {"ArrayLike": "ArrayLike"}

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "JOSS"]

master_doc = "pages/contents"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]}
