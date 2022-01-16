API
===

Computation engine
------------------

.. autoclass:: tinygp.GaussianProcess
   :members:

Kernels
-------

.. autoclass:: tinygp.kernels.Kernel
   :members:

.. autoclass:: tinygp.kernels.Custom
.. autoclass:: tinygp.kernels.AffineTransform
.. autoclass:: tinygp.kernels.SubspaceTransform

.. autoclass:: tinygp.kernels.Sum
.. autoclass:: tinygp.kernels.Product
.. autoclass:: tinygp.kernels.Constant

.. autoclass:: tinygp.kernels.DotProduct
.. autoclass:: tinygp.kernels.Polynomial
.. autoclass:: tinygp.kernels.Linear

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

.. autofunction:: tinygp.metrics.unit_metric
.. autofunction:: tinygp.metrics.diagonal_metric
.. autofunction:: tinygp.metrics.dense_metric
.. autofunction:: tinygp.metrics.cholesky_metric
.. autofunction:: tinygp.metrics.compose


Mean functions
--------------

.. autofunction:: tinygp.means.zero_mean
.. autofunction:: tinygp.means.constant_mean
