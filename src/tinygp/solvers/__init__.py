"""
In ``tinygp``, "solvers" provide a swappable low-level interface for
implementing the linear algebra required to execute Gaussian Process models. At
the moment, ``tinygp`` includes two solvers, but new solvers can be implemented
as external packages or as pull requests to the main ``tinygp`` GitHub project.
The two built in solvers are:

1. :class:`DirectSolver`: A solver that uses a naive approach to solving the
   required linear systems. This can still be pretty fast if you have access to
   a GPU that can run ``jax`` code. This is the default solver, and it can be
   used with any kernel implemented by ``tinygp``. Up to numerical precision,
   this is an *exact* solver.

2. :class:`QuasisepSolver`: An experiemental scalable solver that exploits the
   "quasiseparable" structure found in many GP covariance matrices to make the
   required linear algabra possible in linear scaling with the size of the
   dataset. These methods were previously implemented as part of the `celerite
   <https://celerite.readthedocs.io>`_ project.

Users generally won't instantiate these solvers directly and ``tinygp`` should
generally be able to figure out the best one to use. But you can use a specific
solver using the ``solver`` argument to :class:`tinygp.GaussianProcess` as
follows:

.. code-block:: python

    gp = tinygp.GaussianProcess(..., solver=tinygp.solvers.DirectSolver)

The details for the included solvers are given below, but this is a pretty
low-level feature and the details are definitely subject to change!
"""

__all__ = ["DirectSolver", "QuasisepSolver"]

from tinygp.solvers.direct import DirectSolver
from tinygp.solvers.quasisep import QuasisepSolver
