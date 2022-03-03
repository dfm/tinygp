.. _api-kernels:

kernels package
===============

.. currentmodule:: tinygp.kernels

.. automodule:: tinygp.kernels

.. autosummary::
   :toctree: summary

    Kernel
    Conditioned
    Custom
    Sum
    Product
    Constant
    DotProduct
    Polynomial


.. _stationary-kernels:

Stationary Kernels
------------------

.. automodule:: tinygp.kernels.stationary

.. autosummary::
   :toctree: summary

    Stationary
    Exp
    ExpSquared
    Matern32
    Matern52
    Cosine
    ExpSineSquared
    RationalQuadratic


.. _distance-metrics:

Distance Metrics
----------------

.. automodule:: tinygp.kernels.distance

.. autosummary::
   :toctree: summary

    Distance
    L1Distance
    L2Distance


Subpackages
-----------

.. toctree::
   :maxdepth: 1

   kernels.quasisep
