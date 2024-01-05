.. _release:

Release Notes
=============

.. towncrier release notes start

tinygp 0.3.0 (2024-01-05)
-------------------------

Features
~~~~~~~~

- Added a more robust and better tested implementation of the ``CARMA`` kernel for
  use with the ``QuasisepSolver``. (`#90 <https://github.com/dfm/tinygp/issues/90>`_)
- Switched all base classes to `equinox.Module <https://docs.kidger.site/equinox/api/module/module/>`_ objects to simplify dataclass handling. (`#200 <https://github.com/dfm/tinygp/issues/200>`_)


Bugfixes
~~~~~~~~

- Fixed use of `jnp.roots` and `np.roll` to make CARMA kernel jit-compliant. (`#188 <https://github.com/dfm/tinygp/issues/188>`_)


tinygp 0.2.4 (2023-09-29)
-------------------------

Features
~~~~~~~~

- Removed `__post_init__` checks after kernel construction to avoid extraneous errors when returning kernels out of `jax.vmap`'d functions. (`#148 <https://github.com/dfm/tinygp/issues/148>`_)
- Added Zenodo data to improve citation tracking. (`#151 <https://github.com/dfm/tinygp/issues/151>`_)


Bugfixes
~~~~~~~~

- Fixed syntax for `vmap` of `flax` modules in `transforms` tutorial. (`#159 <https://github.com/dfm/tinygp/issues/159>`_)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Fixed incorrect definition of "spectral mixture kernel" in the custom kernels
  tutorial. (`#143 <https://github.com/dfm/tinygp/issues/143>`_)
- Unpinned the docs theme version to fix release compatibility with recent
  versions of setuptools. (`#153 <https://github.com/dfm/tinygp/issues/153>`_)
- Added past contributor metadata to `.zenodo.json`. (`#154 <https://github.com/dfm/tinygp/issues/154>`_)
- Clarified in documentation that sigma argument is optional in quasisep kernels. (`#176 <https://github.com/dfm/tinygp/issues/176>`_)


Misc
~~~~

- `#184 <https://github.com/dfm/tinygp/issues/184>`_


tinygp 0.2.3 (2022-10-31)
-------------------------

Features
~~~~~~~~

- Removed deprecation warning from ``predict`` method and wrapped it in a
  ``jax.jit`` in order to support interactive use. (`#120 <https://github.com/dfm/tinygp/issues/120>`_)
- Added check for sorted input coordinates when using the ``QuasisepSolver``;
  a ``ValueError`` is thrown if they are not. (`#123 <https://github.com/dfm/tinygp/issues/123>`_)


Bugfixes
~~~~~~~~

- Fixed incorrect definition of ``observation_model`` for ``Celerite`` kernel. (`#88 <https://github.com/dfm/tinygp/issues/88>`_)
- Fixed ``FutureWarning`` by updating ``tree_map`` to ``tree_util.tree_map``. (`#114 <https://github.com/dfm/tinygp/issues/114>`_)
- Fixed issue when tree structure and shape of ``X_test`` input to ``condition``
  was incompatible with the initial input. (`#119 <https://github.com/dfm/tinygp/issues/119>`_)
- Fixed bug where the gradient of the L2 distance would return NaN when the
  distance was zero. (`#121 <https://github.com/dfm/tinygp/issues/121>`_)
- Fixed behavior of DotProduct kernel on scalar inputs. (`#124 <https://github.com/dfm/tinygp/issues/124>`_)


tinygp 0.2.2 (2022-04-20)
-------------------------

Bugfixes
~~~~~~~~

- Fixed dangling ``numpy`` operation in quasiseparable tree map. (`#81 <https://github.com/dfm/tinygp/issues/81>`_)


tinygp 0.2.1 (2022-03-28)
-------------------------

Features
~~~~~~~~

- Renamed elements of quasiseparable kernels, and added support for modeling
  derivative observations with these kernels. (`#58 <https://github.com/dfm/tinygp/issues/58>`_)
- Added more flexible noise models: diagonal, banded, or dense. (`#59 <https://github.com/dfm/tinygp/issues/59>`_)
- Added :class:`tinygp.kernels.quasisep.CARMA` kernel to implement CARMA models. (`#60 <https://github.com/dfm/tinygp/issues/60>`_)
- Added a minimal solver based on Kalman filtering to use as a baseline for
  checking the performance of the :class:`tinygp.solvers.QuasisepSolver`. (`#67 <https://github.com/dfm/tinygp/issues/67>`_)


Bugfixes
~~~~~~~~

- Fixed exception when conditioning with quasiseparable solver, since quasisep
  kernels are not hashable. (`#57 <https://github.com/dfm/tinygp/issues/57>`_)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Added a new tutorial describing how to model multiband and derivative
  observations using quasiseparable kernels. (`#58 <https://github.com/dfm/tinygp/issues/58>`_)
- Add more details to Deep Kernel learning tutorial,
  showing comparison with Matern-3/2 kernel
  and the transformed features. (`#70 <https://github.com/dfm/tinygp/issues/70>`_)


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
