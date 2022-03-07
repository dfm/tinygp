.. _api-ref:

Public API
==========

The following pages describe the technical details of all the public-facing
members of the ``tinygp`` API. This isn't meant to be introductory and, if
you're new here, the :ref:`tutorials` might be a better place to start. That
being said, we've tried to provide sufficiently detailed descriptions of all the
provided methods for once you (/we) get into the weeds. Please `open issues or
pull requests <https://github.com/dfm/tinygp/issues>`_ if you find anything
lacking.

Primary Interface
-----------------

.. currentmodule:: tinygp

.. automodule:: tinygp

.. autosummary::
   :toctree: summary

   GaussianProcess
   gp.ConditionResult


Subpackages
-----------

.. toctree::
    :maxdepth: 1

    kernels
    means
    noise
    solvers
    transforms
