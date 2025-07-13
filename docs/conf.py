import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../src'))

# Copy notebooks into docs/examples before build
examples_src = os.path.abspath('../examples')
examples_dst = os.path.abspath('examples')
shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pcxarray'
copyright = '2025, Mississippi State University'
author = "GCER Lab @ Mississippi State University"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/Numpy style docstrings
    'numpydoc',
    'nbsphinx',  
    'myst_parser', 
]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = '_static/gcer_alt.png'
