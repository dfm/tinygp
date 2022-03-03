.. _release:

Release Notes
=============

.. towncrier release notes start

tinygp 0.2.0 (2022-03-03)
-------------------------

Features
~~~~~~~~

- Added new interface for conditioning GP models. ``condition`` method now returns
  a :class:`tinygp.GaussianProcess` object describing the conditional
  distribution. (`#32 <https://github.com/dfm/tinygp/issues/32>`_)
- Added new experimental scalable solver using quasiseparable matrices. See
  :ref:`api-kernels-quasisep`, :ref:`api-solvers-quasisep`, and
  :class:`tinygp.solvers.quasisep.solver.QuasisepSolver` for more information. (`#47 <https://github.com/dfm/tinygp/issues/47>`_)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Updated benchmarks to include quasiseparable solver. (`#49 <https://github.com/dfm/tinygp/issues/49>`_)
- Major overhaul of API documentation. Added many docstrings and expanded text
  thoughout the API docs pages. (`#52 <https://github.com/dfm/tinygp/issues/52>`_)
- Added 3 new tutorials: (1) :ref:`intro`, giving a general introduction to
  ``tinygp``, (2) :ref:`means`, showing how ``tinygp`` can be used with a
  non-trivial mean function, and (3) :ref:`quasisep`, introducing the scalable
  solver for quasiseparable kernels. (`#54 <https://github.com/dfm/tinygp/issues/54>`_)
- Added support for `towncrier <https://github.com/twisted/towncrier>`_ generated
  release notes. (`#55 <https://github.com/dfm/tinygp/issues/55>`_)


Deprecations and Removals
~~~~~~~~~~~~~~~~~~~~~~~~~

- Breaking change: Removed existing ``condition`` method and deprected ``predict``
  method. (`#32 <https://github.com/dfm/tinygp/issues/32>`_)
