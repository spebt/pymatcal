# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import os
# import sys
import pymatcal
project = "pymatcal"
copyright = "2024, Fang Han"
author = "Fang Han"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# sys.path.insert(0, os.path.abspath("../.."))
# sys.path.insert(0, os.path.abspath("../../pymatcal"))
# sys.path.insert(0, os.path.abspath("."))
# print(os.path)

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "myst_parser",
    'sphinx_toolbox.assets',
    "sphinx_design"
]

myst_enable_extensions = ["colon_fence"]
# autosummary_generate = True
# Add mappings
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}
templates_path = ["_templates"]
exclude_patterns = []
mater_doc = "toc"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "about.html",
        "searchfield.html",
        "navigation.html",
        "relations.html",
        "donate.html",
    ]
}

html_theme_options = {
    "github_user": "spebt",
    "github_repo": "pymatcal",
    "github_button": True,
    "show_relbars": True,
}
