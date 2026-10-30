# -*- coding: utf-8 -*-
"""Sphinx documentation configuration for MNESIS.

Sources live in ``docs/`` : Markdown landing pages and symlinks to the
Jupyter notebooks of ``src/`` (created by ``make notebooks``) are built as
first-class doc pages, while the core code of ``src/mnesis_chains.py`` and
``src/mnesis_boilerplate.py`` is rendered from its docstrings via autodoc.
The built HTML is deployed to GitHub Pages by a GitHub Actions workflow.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)   # so that relative paths ('../cached_data', '../figures') resolve as in src/


# -- Project information ------------------------------------------------------
project = "MNESIS"
copyright = "2026, Laurent U Perrinet"
author = "Laurent U Perrinet"

version_raw = "unknown"
try:
    from mnesis_boilerplate import datetag as version_raw   # single source of truth
except Exception:
    pass

# Pages build passes `_RTD_VERSION` to avoid caching stale values across branches.
version = os.environ.get("_RTD_VERSION", version_raw)


# -- General Sphinx settings --------------------------------------------------
extensions = [
    "myst_parser",                  # Markdown (and Markdown-like notebooks)
    "nbsphinx",                     # Jupyter notebook rendering + output embedding
    "sphinx.ext.autodoc",           # API pages from docstrings
    "sphinx.ext.napoleon",          # Google-style docstrings
    "sphinx.ext.intersphinx",
]

exclude_patterns = [
    "_build",
    ".venv/",
    "cached_data/*",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "html_admonition",
    "dollarmath",
    "amsmath",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_mock_imports = []           # real torch/snntorch are installed for the build
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}


# -- HTML output ---------------------------------------------------------------
templates_path = []

html_theme = "furo"
html_static_path = []
html_extra_path = ["izhikevich.html"]   # interactive raster, embedded in index.md
html_title = "MNESIS | Working Memory with Polychronous Chains"

html_context = {
    "docs_github_url": "https://spikeai.github.io/MNESIS/",
}

html_theme_options = {
    "source_repository": "https://github.com/SpikeAI/MNESIS/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# -- nbsphinx rendering --------------------------------------------------------
nbsphinx_execute = "never"           # render stored outputs; skip execution during CI
nbsphinx_execute_arguments = [
    "--InlineMathPlugin.enable",
]
nbsphinx_allow_errors = True
