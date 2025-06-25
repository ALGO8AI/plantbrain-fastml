# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# This line adds the project's root directory to Python's path.
# From conf.py, '..' goes to 'docs', and another '..' goes to the root.
sys.path.insert(0, os.path.abspath('../..'))


project = 'PLANTBRAIN-FASTML'
copyright = '2025, Himanshu Ranjan, Himanshu Bhansali'
author = 'Himanshu Ranjan, Himanshu Bhansali'
release = '0.4.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

