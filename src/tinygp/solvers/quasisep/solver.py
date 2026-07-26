from __future__ import annotations

__all__ = ["QuasisepSolver"]

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tinygp import means
from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise
from tinygp.solvers.quasisep import ops, predict as qsp
from tinygp.solvers.quasisep.core import (
    DiagQSM,
    LowerTriQSM,
    StrictLowerTriQSM,
    SymmQSM,
)
from tinygp.solvers.solver import (
    ConditionedComponents,
    Solver,
    conditioned_mean_parts,
)

if TYPE_CHECKING:
    from tinygp.kernels.quasisep import Quasisep


class QuasisepSolver(Solver):
    """A scalable solver that uses quasiseparable matrices

    Take a look at the documentation for the :ref:`api-solvers-quasisep`, for
    more technical details.

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`QuasisepSolver.init` method instead of the
    usual constructor.
    """

    X: JAXArray
    kernel: Kernel
    matrix: SymmQSM
    factor: LowerTriQSM
    parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
        assume_sorted: bool = False,
        parallel: bool = False,
    ):
        """Build a :class:`QuasisepSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed
                :class:`tinygp.solvers.quasisep.core.QSM` with the covariance
                matrix.
            assume_sorted: If ``True``, assume that the input coordinates are
                sorted. If ``False``, check that they are sorted and throw an
                error if they are not. This can introduce a runtime overhead,
                and you can pass ``assume_sorted=True`` to get the best
                performance.
            parallel: If ``True``, use parallel associative-scan algorithms for
                the Cholesky factorization, triangular solves, and matrix
                products. This trades increased FLOPs for reduced sequential
                depth and can be substantially faster on GPUs/TPUs for large
                ``N``.
        """
        if covariance is None:
            if TYPE_CHECKING:
                assert isinstance(kernel, Quasisep)
            if not assume_sorted:
                jax.debug.callback(_check_sorted, kernel.coord_to_sortable(X))
            matrix = kernel.to_symm_qsm(X)
            matrix += noise.to_qsm()
        else:
            if TYPE_CHECKING:
                assert isinstance(covariance, SymmQSM)
            matrix = covariance

        self.X = X
        self.kernel = kernel
        self.matrix = matrix
        self.parallel = parallel
        (d,) = matrix.diag
        p, q, a = matrix.lower
        impl = ops.cholesky_parallel if parallel else ops.cholesky
        c, w = impl(d, p, q, a)
        self.factor = LowerTriQSM(
            diag=DiagQSM(c), lower=StrictLowerTriQSM(p=p, q=w, a=a)
        )

    def variance(self) -> JAXArray:
        return self.matrix.diag.d

    def covariance(self) -> JAXArray:
        return self.matrix.to_dense()

    def normalization(self) -> JAXArray:
        return jnp.sum(jnp.log(self.factor.diag.d)) + 0.5 * self.factor.shape[
            0
        ] * np.log(2 * np.pi)

    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        if transpose:
            return self.factor.transpose().solve(y, parallel=self.parallel)
        else:
            return self.factor.solve(y, parallel=self.parallel)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self.factor.matmul(y, parallel=self.parallel)

    def condition(
        self,
        kernel: Kernel | None,
        X_train: JAXArray,
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
        *,
        include_mean: bool,
        mean_function: means.MeanBase,
    ) -> ConditionedComponents:
        """Build the components of a conditioned GP.

        Three regimes:

        - ``X_test is not None`` and ``kernel is None`` (the user did not
          override the kernel): the fast O(log N + J^2)-per-test-point path from
          :mod:`tinygp.solvers.quasisep.predict` for the mean and diagonal
          variance, with a lazy ``ConditionedSolver`` that builds the dense
          covariance only on demand. Gating on ``kernel is None`` rather than
          ``kernel is self.kernel`` is what keeps this path live under
          ``jax.jit`` (object identity is not preserved across the trace).
        - ``X_test is None`` with a :class:`~tinygp.kernels.quasisep.Quasisep`
          kernel: the conditional covariance is computed as a QSM at the
          training coordinates.
        - Otherwise (a cross-kernel was supplied, or a non-Quasisep kernel): a
          dense covariance.
        """
        # Imported here, not at module scope: ``tinygp.kernels.quasisep`` imports
        # the quasiseparable core/block, whose package import pulls in this
        # solver, so a top-level import would be circular. It is needed at
        # runtime for the ``isinstance(kernel, Quasisep)`` check below.
        from tinygp.kernels.quasisep import Quasisep

        # Fast path: predicting at test points with the training kernel. The
        # QuasisepSolver always holds a Quasisep kernel, so we only need to check
        # that the user did not pass a (cross-)kernel.
        if X_test is not None and kernel is None:
            pred_kernel = self.kernel
            state = qsp.precompute(self, alpha, parallel=self.parallel)
            cond_mean = qsp.ConditionedMean(
                state=state,
                kernel=pred_kernel,
                include_mean=include_mean,
                mean_function=mean_function,
            )
            cond_kernel = qsp.ConditionedKernel(X_train, self, pred_kernel, state)

            # One shared anchor pass per test point gives both mean and variance.
            raw_mean, raw_var = jax.vmap(
                lambda x: qsp.predict_mean_and_var(pred_kernel, state, x)
            )(X_test)
            mean_value = raw_mean
            if include_mean:
                mean_value = mean_value + jax.vmap(mean_function)(X_test)
            var = raw_var + noise.diagonal()
            return ConditionedComponents(
                mean=cond_mean,
                kernel=cond_kernel,
                mean_value=mean_value,
                variance_value=var,
                covariance_value=qsp.ConditionedCovariance(
                    train_solver=self, train_kernel=pred_kernel
                ),
            )

        kernel = self.kernel if kernel is None else kernel
        cond_mean, cond_kernel, mean_value = conditioned_mean_parts(
            self,
            kernel,
            X_train,
            X_test,
            alpha,
            include_mean=include_mean,
            mean_function=mean_function,
        )

        # We can easily compute the conditional as a QSM in the special case
        # where we are predicting at the input coordinates and a Quasisep kernel
        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(X_train)
            delta = (self.factor.inv() @ M).gram()
            M += noise.to_qsm()
            covariance_value: Any = M - delta
        else:
            if X_test is None:
                Kss = Ks = kernel(X_train, X_train)
            else:
                Kss = kernel(X_test, X_test)
                Ks = kernel(X_train, X_test)
            A = self.solve_triangular(Ks)
            covariance_value = Kss - A.transpose() @ A + noise

        return ConditionedComponents(
            mean=cond_mean,
            kernel=cond_kernel,
            mean_value=mean_value,
            variance_value=None,
            covariance_value=covariance_value,
        )


def _check_sorted(X: JAXArray) -> None:
    if np.any(np.diff(X) < 0.0):
        raise ValueError(
            "Input coordinates must be sorted in order to use the QuasisepSolver"
        )
