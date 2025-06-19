# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath('../../src'))

project = 'ConfigurableOptimizer'
copyright = '2025, Arjun Krishnakumar, Shakiba Moradian, Abhash Kumar Jha'
author = 'Arjun Krishnakumar, Abhash Kumar Jha, Shakiba Moradian'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns: list[Any] = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  # Provided by Google in your dashboard
    'logo_only': False,
    # 'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'myst_parser',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Disable if you're only using Google style

napoleon_use_param = True
napoleon_use_ivar = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_rtype = False  # Set to False so the return type is in the same line as the description