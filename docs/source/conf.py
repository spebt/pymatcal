# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pymatcal

project = "pymatcal"
copyright = "2024, Fang Han"
author = "Fang Han"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
	"ablog",
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Add mappings
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}

myst_words_per_minute = 200
templates_path = ["_templates"]
exclude_patterns = []
html_last_updated_fmt = "%b %d, %Y"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_prev_next": False,
    "secondary_sidebar_items": [
        "page-toc",
        "edit-this-page",
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "navbar_align": "right",
    "content_footer_items": [],
    "footer_start": ["version", "last-updated"],
    "footer_center": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "back_to_top_button": True,
    "icon_links": [
        {
            # Label for this link
            "name": "SPEBT GitHub PyMatCal",
            # URL where the link will redirect
            "url": "https://github.com/spebt/pymatcal/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "logo": {
        "link": "https:/spebt.github.io/",
        "image_light": "_static/img/logo-light.png",
        "image_dark": "_static/img/logo-dark.png",
        "text": "Documentation",
        "alt_text": "SPEBT Project Documentation - Home",
    },
}
html_sidebars = {
    "quickstart": [],
    "guides/*": [],
    "api-ref": [],
	"INSTALL": [],
    
}
