# -*- coding: utf-8 -*-

__all__ = ["__version__", "kernels", "transforms", "GaussianProcess"]

from tinygp import kernels, transforms
from tinygp.gp import GaussianProcess
from tinygp.tinygp_version import version as __version__

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://github.com/dfm/tinygp"
__license__ = "MIT"
__description__ = "The tiniest of Gaussian Process libraries"
__copyright__ = "2021, 2022 Simons Foundation, Inc."
__contributors__ = "https://github.com/dfm/tinygp/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
