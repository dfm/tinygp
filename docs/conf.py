# -*- coding: utf-8 -*-

import tinygp

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# General information about the project.
project = "tinygp"
copyright = tinygp.__copyright__
version = tinygp.__version__
release = tinygp.__version__

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "tinygp"
html_logo = "_static/zap.svg"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dfm/tinygp",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://tinygp.readthedocs.io/en/latest/"
jupyter_execute_notebooks = "auto"
execution_excludepatterns = ["tutorials/george.ipynb"]
execution_timeout = -1


autodoc_type_aliases = {
    "JAXArray": "tinygp.types.JAXArray",
    "Axis": "tinygp.kernels.Axis",
    "Mean": "tinygp.means.Mean",
    "Metric": "tinygp.metrics.Metric",
}
