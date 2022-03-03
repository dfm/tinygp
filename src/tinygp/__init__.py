# -*- coding: utf-8 -*-
"""
``tinygp`` is an extremely lightweight library for building Gaussian Process
models in Python, built on top of `jax <https://github.com/google/jax>`_. The
primary way that you will use to interact with ``tinygp`` is by constructing
"kernel" functions using the building blocks provided in the ``kernels``
subpackage (see :ref:`api-kernels`), and then passing that to a
:class:`GaussianProcess` object to do all the computations. Check out the
:ref:`tutorials` for a more complete introduction.
"""

__all__ = ["kernels", "solvers", "transforms", "GaussianProcess"]

from tinygp import kernels, solvers, transforms
from tinygp.gp import GaussianProcess
from tinygp.tinygp_version import version as __version__

__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__uri__ = "https://github.com/dfm/tinygp"
__license__ = "MIT"
__description__ = "The tiniest of Gaussian Process libraries"
__copyright__ = "2021, 2022 Simons Foundation, Inc"
__contributors__ = "https://github.com/dfm/tinygp/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
