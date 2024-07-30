# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
project = 'pymatcal'
copyright = '2024, Fang Han'
author = 'Fang Han'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../pymatcal'))
sys.path.insert(0, os.path.abspath('.'))
# print(os.path)

extensions = ['sphinx.ext.intersphinx','sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'myst_parser']
# autosummary_generate = True
# Add mappings
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'python': ('https://docs.python.org/3', None),
}
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
