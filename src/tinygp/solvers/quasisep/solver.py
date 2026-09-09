from __future__ import annotations

__all__ = ["QuasisepSolver"]

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Conditioned, Kernel
from tinygp.noise import Noise
from tinygp.solvers.direct import dense_condition
from tinygp.solvers.quasisep.core import (
    DiagQSM,
    LowerTriQSM,
    StrictLowerTriQSM,
    SymmQSM,
)
from tinygp.solvers.solver import ConditionedComponents, Solver


class QuasisepSolver(Solver):
    """A scalable solver that uses quasiseparable matrices

    Take a look at the documentation for the :ref:`api-solvers-quasisep`, for
    more technical details.

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`QuasisepSolver.init` method instead of the
    usual constructor.
    """

    kernel: Kernel
    X: JAXArray
    noise: Noise
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
        from tinygp.kernels.quasisep import Quasisep

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
        self.kernel = kernel
        self.X = X
        self.noise = noise
        self.matrix = matrix
        self.parallel = parallel
        self.factor = matrix.cholesky(parallel=parallel)

    def variance(self) -> JAXArray:
        return self.matrix.diag.d

    def covariance(self) -> JAXArray:
        N = self.matrix.shape[0]
        return self.matrix.matmul(jnp.eye(N), parallel=self.parallel)

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

    def _conditional_at_data(self) -> SymmQSM:
        """The conditional covariance at the input coordinates, as a QSM

        When conditioning on the observed data using the same kernel that
        this solver was built with, and with ``K = M + N`` the full covariance
        matrix, the conditional covariance ``M - M @ K^{-1} @ M`` simplifies
        to ``N - N @ K^{-1} @ N``. This form only requires the inverse of
        ``K``, and it has the same quasiseparable rank as the original
        kernel. It is also much better conditioned than computing
        ``M - M @ K^{-1} @ M`` directly, where the difference of two nearly
        equal matrices is encoded in the generators.
        """
        from tinygp.solvers.quasisep.ops import qsm_mul

        Kinv = self.matrix.inv(parallel=self.parallel)
        N = self.noise.to_qsm()
        if isinstance(N, DiagQSM):
            n = N.d
            lam = Kinv.diag.d
            t, s, ell = Kinv.lower
            return SymmQSM(
                diag=DiagQSM(d=n - jnp.square(n) * lam),
                lower=StrictLowerTriQSM(p=-n[:, None] * t, q=n[:, None] * s, a=ell),
            )

        P = qsm_mul(N, qsm_mul(Kinv, N, parallel=self.parallel), parallel=self.parallel)
        return N - SymmQSM(diag=P.diag, lower=P.lower)

    def _conditional_delta(self, M: SymmQSM) -> SymmQSM:
        # The (QSM) term M @ K^{-1} @ M = (L^{-1} @ M)^T @ (L^{-1} @ M) that
        # gets subtracted from M when conditioning at the input coordinates
        # with a general (cross-)kernel M
        from tinygp.solvers.quasisep.ops import qsm_mul

        A = qsm_mul(self.factor.inv(), M, parallel=self.parallel)
        return A.gram(parallel=self.parallel)

    def condition(
        self,
        kernel: Kernel | None,
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
    ) -> ConditionedComponents:
        """Build the components of the conditioned process

        When predicting at the input coordinates (``X_test=None``) with a
        :class:`tinygp.kernels.quasisep.Quasisep` kernel, the conditional
        covariance is computed as a quasiseparable matrix, so the conditioned
        process is itself a ``QuasisepSolver`` process. When, in addition, the
        kernel is the one that this solver was built with (``kernel=None``), a
        rank-``J`` representation is used that only requires the inverse of the
        training covariance and is numerically much better behaved (see
        :func:`_conditional_at_data`). That applies to any ``QuasisepSolver``,
        including one that is itself the result of conditioning.

        Otherwise, this falls back on
        :func:`tinygp.solvers.direct.dense_condition`, which materializes a
        dense conditional covariance, so be careful when predicting at a large
        number of test points!
        """
        from tinygp.kernels.quasisep import Quasisep

        pred_kernel = self.kernel if kernel is None else kernel

        if X_test is not None:
            comps = dense_condition(self, kernel, X_test, noise, alpha)
            if isinstance(pred_kernel, Quasisep):
                # The cross-covariance matmul is O((N + M) J^2) for a Quasisep
                # kernel, so prefer it to the dense product from ``Ks``
                mean_value = pred_kernel.matmul(X_test, self.X, alpha)
                comps = comps._replace(mean_value=mean_value)
            return comps

        if kernel is None:
            covariance = self._conditional_at_data() + noise.to_qsm()
            # The conditional mean at the data is M @ alpha = (K - N) @ alpha
            # with K = M + N the full training covariance; unlike the kernel's
            # matmul, this is cheap for any kernel, including the
            # ``Conditioned`` kernel of an already conditioned process
            mean_value = self.matrix.matmul(alpha, parallel=self.parallel)
            mean_value -= self.noise @ alpha
        elif isinstance(pred_kernel, Quasisep):
            M = pred_kernel.to_symm_qsm(self.X)
            covariance = M + noise.to_qsm() - self._conditional_delta(M)
            mean_value = pred_kernel.matmul(self.X, y=alpha)
        else:
            return dense_condition(self, kernel, X_test, noise, alpha)

        cond_kernel = Conditioned(self.X, self, pred_kernel)
        return ConditionedComponents(
            kernel=cond_kernel,
            mean_value=mean_value,
            solver=QuasisepSolver(
                cond_kernel,
                self.X,
                noise,
                covariance=covariance,
                parallel=self.parallel,
            ),
        )


def _check_sorted(X: JAXArray) -> None:
    if np.any(np.diff(X) < 0.0):
        raise ValueError(
            "Input coordinates must be sorted in order to use the QuasisepSolver"
        )
