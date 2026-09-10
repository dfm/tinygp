Added a fast prediction path for the ``QuasisepSolver``: conditioning at new
test points with the kernel used for fitting (``kernel=None``, the default)
now evaluates the predictive variance in ``O(J^2)`` per test point by reusing
the quasiseparable Cholesky factorization, and the conditioned
``GaussianProcess`` only builds its dense conditional covariance when it is
explicitly requested (``covariance``, ``sample``, or ``log_probability``).
Passing any ``kernel`` argument to ``condition``/``predict``, even the training
kernel itself, uses the dense path instead. The cross-covariance products used
for the predictive mean now handle far extrapolation without ``NaN``
gradients. ``GaussianProcess`` also accepts an already constructed solver
instance via its ``solver`` argument.
