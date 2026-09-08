from __future__ import annotations

__all__ = ["QuasisepSolver"]

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise
from tinygp.solvers.quasisep.core import (
    DiagQSM,
    LowerTriQSM,
    StrictLowerTriQSM,
    SymmQSM,
)
from tinygp.solvers.solver import Solver


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
        self, kernel: Kernel | None, X_test: JAXArray | None, noise: Noise
    ) -> Any:
        """Compute the covariance matrix for a conditional GP

        In the case where the prediction is made at the input coordinates with a
        :class:`tinygp.kernels.quasisep.Quasisep` kernel, this will return the
        quasiseparable representation of the conditional matrix. Otherwise, it
        will use scalable methods where possible, but return a dense
        representation of the covariance, so be careful when predicting at a
        large number of test points!

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data. If ``None``, the kernel used to construct this
                solver is used, which enables a more efficient and numerically
                stable algorithm when ``X_test`` is also ``None``.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
        """
        from tinygp.kernels.quasisep import Quasisep

        # The most common case: predicting at the input coordinates with the
        # kernel that this solver was built with
        if X_test is None and kernel is None:
            return self._conditional_at_data() + noise.to_qsm()

        if kernel is None:
            kernel = self.kernel

        # We can also compute the conditional as a QSM in the special case
        # where we are predicting at the input coordinates and a Quasisep kernel
        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(self.X)
            delta = self._conditional_delta(M)
            M += noise.to_qsm()
            return M - delta

        # Otherwise fall back on the slow method for now :(
        if X_test is None:
            Kss = Ks = kernel(self.X, self.X)
        else:
            Kss = kernel(X_test, X_test)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A

    def condition_diag(
        self, kernel: Kernel | None, X_test: JAXArray | None, noise: Noise
    ) -> JAXArray:
        """The diagonal of the covariance matrix for a conditional GP

        This reuses the same quasiseparable special cases as :func:`condition`:
        when predicting at the input coordinates with a
        :class:`tinygp.kernels.quasisep.Quasisep` kernel, the diagonal can be
        computed in ``O(N)`` (or ``O(N log N)`` with the parallel algorithms)
        without ever materializing a dense matrix. Otherwise, this falls back on
        :func:`tinygp.solvers.solver.Solver.condition_diag`.
        """
        from tinygp.kernels.quasisep import Quasisep

        if X_test is None and kernel is None:
            return self._conditional_at_data().diag.d + noise.diagonal()

        if kernel is None:
            kernel = self.kernel

        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(self.X)
            delta = self._conditional_delta(M)
            M += noise.to_qsm()
            return M.diag.d - delta.diag.d

        return super().condition_diag(kernel, X_test, noise)


def _check_sorted(X: JAXArray) -> None:
    if np.any(np.diff(X) < 0.0):
        raise ValueError(
            "Input coordinates must be sorted in order to use the QuasisepSolver"
        )
