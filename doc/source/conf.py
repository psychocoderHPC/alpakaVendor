# Configuration file for the Sphinx documentation builder.

import os
import sys

# allow importing sphinx_helper modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sphinx_helper.doxygen import generate_doxygen


def setup(app):
    app.connect("builder-inited", generate_doxygen)


# -- Project information -----------------------------------------------------

project = "alpakaVendor"
copyright = "Documentation under CC-BY 4.0"
author = "The alpakaVendor team."
version = "0.1.0"
release = "0.1.0"

master_doc = "index"

# -- General configuration ---------------------------------------------------

highlight_language = "c++"

extensions = [
    "sphinx.ext.mathjax",
    "breathe",
    "sphinxcontrib.programoutput",
    "sphinx.ext.autosectionlabel",
]
autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]
source_suffix = [".rst"]
language = "en"
pygments_style = "sphinx"
todo_include_todos = False

# -- Breathe configuration --------------------------------------------------

breathe_projects = {"alpakaVendor": "../doxygen/xml"}
breathe_default_project = "alpakaVendor"
breathe_domain_by_extension = {"cpp": "cpp", "h": "cpp", "hpp": "cpp"}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": r"\setcounter{tocdepth}{2}",
}

latex_documents = [
    (master_doc, "alpakaVendor-doc.tex", "alpakaVendor Documentation", "The alpakaVendor Community", "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "alpakaVendor", "alpakaVendor Documentation", [author], 1)]
