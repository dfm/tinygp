"""
``tinygp`` is an extremely lightweight library for building Gaussian Process
models in Python, built on top of `jax <https://github.com/google/jax>`_. The
primary way that you will use to interact with ``tinygp`` is by constructing
"kernel" functions using the building blocks provided in the ``kernels``
subpackage (see :ref:`api-kernels`), and then passing that to a
:class:`GaussianProcess` object to do all the computations. Check out the
:ref:`tutorials` for a more complete introduction.
"""

from tinygp import (
    kernels as kernels,
    noise as noise,
    solvers as solvers,
    transforms as transforms,
)
from tinygp.gp import GaussianProcess as GaussianProcess
from tinygp.tinygp_version import __version__ as __version__
