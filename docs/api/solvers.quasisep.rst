.. _api-solvers-quasisep:

solvers.quasisep package
========================

.. currentmodule:: tinygp.solvers.quasisep

.. automodule:: tinygp.solvers.quasisep


Square Quasiseparable Matrices
------------------------------

.. currentmodule:: tinygp.solvers.quasisep.core

.. automodule:: tinygp.solvers.quasisep.core

.. autosummary::
   :toctree: summary

   QSM
   DiagQSM
   StrictLowerTriQSM
   StrictUpperTriQSM
   LowerTriQSM
   UpperTriQSM
   SquareQSM
   SymmQSM


Rectangular Quasiseparable Matrices
-----------------------------------

.. currentmodule:: tinygp.solvers.quasisep.general

.. automodule:: tinygp.solvers.quasisep.general

.. autosummary::
   :toctree: summary

   GeneralQSM


Fast Prediction
---------------

.. currentmodule:: tinygp.solvers.quasisep.predict

.. automodule:: tinygp.solvers.quasisep.predict

.. autosummary::
   :toctree: summary

   PredictState
   precompute
   predict_var
   ConditionedKernel
