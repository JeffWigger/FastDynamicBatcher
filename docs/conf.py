# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
# sys.path.insert(0, os.path.abspath("../"))
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

project = "Fast Dynamic Batcher"
copyright = "2024, Jeffrey Wigger"
author = "Jeffrey Wigger"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.duration", "sphinx.ext.doctest", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
