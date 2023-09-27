import tinygp

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]

# General information about the project.
project = "tinygp"
copyright = "2021, 2022, 2023 Simons Foundation, Inc."
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
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["benchmarks.ipynb"]
nb_execution_timeout = -1

autodoc_type_aliases = {
    "JAXArray": "tinygp.helpers.JAXArray",
    "Axis": "tinygp.kernels.Axis",
    "Distance": "tinygp.kernels.distance.Distance",
}
