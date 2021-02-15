# -*- coding: utf-8 -*-

__all__ = ["__version__", "kernels", "means", "GaussianProcess"]

from .tinygp_version import version as __version__
from . import kernels, means
from .gp import GaussianProcess

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://github.com/dfm/tinygp"
__license__ = "MIT"
__description__ = "The tiniest of Gaussian Process libraries"
__copyright__ = "Copyright 2021 Daniel Foreman-Mackey"
__contributors__ = "https://github.com/dfm/tinygp/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
