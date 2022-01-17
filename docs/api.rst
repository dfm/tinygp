.. _api:

API
===

Computation engine
------------------

.. autoclass:: tinygp.GaussianProcess
   :members:

.. _Kernels:

Kernels
-------

.. autoclass:: tinygp.kernels.Kernel
   :members:

.. autoclass:: tinygp.kernels.Custom
.. autoclass:: tinygp.kernels.Transform
.. autoclass:: tinygp.kernels.Subspace

.. autoclass:: tinygp.kernels.Constant
.. autoclass:: tinygp.kernels.DotProduct
.. autoclass:: tinygp.kernels.Polynomial

.. autoclass:: tinygp.kernels.Exp
.. autoclass:: tinygp.kernels.ExpSquared
.. autoclass:: tinygp.kernels.Matern32
.. autoclass:: tinygp.kernels.Matern52
.. autoclass:: tinygp.kernels.Cosine
.. autoclass:: tinygp.kernels.ExpSineSquared
.. autoclass:: tinygp.kernels.RationalQuadratic


.. _Metrics:

Metrics
-------

In ``tinygp``, a "metric" is any callable that takes an input coordinate and
returns a transformed coordinate. There are some built in implementations for
standard linear transformations that can be used to handle multivariate vector
inputs.

.. autofunction:: tinygp.metrics.unit_metric
.. autofunction:: tinygp.metrics.diagonal_metric
.. autofunction:: tinygp.metrics.dense_metric
.. autofunction:: tinygp.metrics.cholesky_metric


Mean functions
--------------

In ``tinygp``, a mean function is specified as a callable that takes an input
coordinate and returns the scalar mean value at that point. This will be
``vmap``-ed, so it should treat its input as a single coordinate and leave
broadcasting to the :class:`tinygp.GaussianProcess` object.

.. autofunction:: tinygp.means.zero_mean
.. autofunction:: tinygp.means.constant_mean
