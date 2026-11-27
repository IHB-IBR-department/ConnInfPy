# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# Make the checkout importable (also for autodoc) regardless of the working
# directory Sphinx is invoked from.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'conninfpy'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Single source of truth for the version: conninfpy/_version.py, the same
# module setuptools reads for the distribution version (pyproject.toml).
from conninfpy._version import __version__

project = 'ConnInfPy'
copyright = '2026, IHB RAS'
author = 'N.P. Bechtereva Institute of the Human Brain, Russian Academy of Sciences - IHB RAS'
release = __version__
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
]
autosummary_generate = True

# Tolerate missing docstring sections under a methods paper deadline.
# Re-enable once the public API reference pass lands.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# Match the theme installed by .github/workflows/documentation.yml so the
# CI-deployed and locally-built docs render the same way.
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'
html_static_path = ['_static']
