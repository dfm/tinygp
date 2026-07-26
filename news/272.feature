Added a fast prediction path for the ``QuasisepSolver``: conditioning at test
points with the training kernel now evaluates the predictive mean and variance
in ``O(J^2)`` per test point by reusing the quasiseparable Cholesky
factorization, instead of building a dense conditional covariance. The fast
path is used whenever the ``kernel`` argument to ``condition``/``predict`` is
omitted; supplying any kernel (even the training kernel) falls back to the
dense path. As part of this change, the signature of the low-level
``Solver.condition`` method changed to return a ``ConditionedComponents``
bundle, which is a breaking change for third-party solver implementations.
