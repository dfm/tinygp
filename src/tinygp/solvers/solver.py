from __future__ import annotations

__all__ = ["Solver", "ConditionedComponents", "conditioned_mean_parts"]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Conditioned as ConditionedKernel, Kernel
from tinygp.means import Conditioned as ConditionedMean
from tinygp.noise import Noise

if TYPE_CHECKING:
    from tinygp.means import MeanBase


class ConditionedComponents(NamedTuple):
    """Everything ``GaussianProcess.condition`` needs to build the conditioned GP.

    Returned by :meth:`Solver.condition`. ``mean_value`` should already include
    the prior mean function when ``include_mean`` was requested;
    ``variance_value`` (if not ``None``) should include the test-side noise.

    ``covariance_value`` is consumed by
    :func:`tinygp.solvers.select_solver` to choose the conditioned GP's solver.
    It may be a dense array (-> :class:`~tinygp.solvers.DirectSolver`), a
    :class:`~tinygp.solvers.quasisep.core.SymmQSM` (-> ``QuasisepSolver``), or a
    solver-specific marker such as ``ConditionedCovariance`` whose presence
    selects a custom solver. Returning an unrecognized type falls back to the
    dense solver, so a custom solver must register its marker in
    ``select_solver``.
    """

    mean: MeanBase
    kernel: Kernel
    mean_value: JAXArray
    variance_value: JAXArray | None
    covariance_value: Any


def conditioned_mean_parts(
    solver: Solver,
    kernel: Kernel,
    X_train: JAXArray,
    X_test: JAXArray | None,
    alpha: JAXArray,
    *,
    include_mean: bool,
    mean_function: MeanBase,
) -> tuple[MeanBase, Kernel, JAXArray]:
    """Build the generic conditioned mean, kernel, and evaluated mean vector.

    Shared by the solvers' fall-back (non-fast) conditioning paths so the
    ``means.Conditioned`` / ``kernels.Conditioned`` construction lives in one
    place. ``kernel`` must already be resolved (not ``None``).
    """
    cond_mean = ConditionedMean(
        X_train,
        alpha,
        kernel,
        include_mean=include_mean,
        mean_function=mean_function,
    )
    cond_kernel = ConditionedKernel(X_train, solver, kernel)
    Xt = X_train if X_test is None else X_test
    mean_value = kernel.matmul(Xt, X_train, alpha)
    if include_mean:
        mean_value = mean_value + jax.vmap(mean_function)(Xt)
    return cond_mean, cond_kernel, mean_value


class Solver(eqx.Module):
    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ):
        del kernel, X, noise, covariance
        raise NotImplementedError

    # TODO(dfm): Add a deprecation warning. This exists for backwards
    # compatibility, but using __init__ directly is preferred.
    @classmethod
    def init(
        cls,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ) -> Solver:
        return cls(kernel, X, noise, covariance=covariance)

    @abstractmethod
    def variance(self) -> JAXArray:
        """The diagonal of the covariance matrix"""
        raise NotImplementedError

    @abstractmethod
    def covariance(self) -> JAXArray:
        """The evaluated covariance matrix"""
        raise NotImplementedError

    @abstractmethod
    def normalization(self) -> JAXArray:
        """The multivariate normal normalization constant

        This should be ``(log_det + n*log(2*pi))/2``, where ``n`` is the size of
        the covariance matrix, and ``log_det`` is the log determinant of the
        matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        """Solve the lower triangular linear system defined by this solver

        If the covariance matrix is ``K = L @ L.T`` for some lower triangular
        matrix ``L``, this method solves ``L @ x = y`` for some ``y``. If the
        ``transpose`` parameter is ``True``, this instead solves ``L.T @ x =
        y``.
        """
        raise NotImplementedError

    @abstractmethod
    def dot_triangular(self, y: JAXArray) -> JAXArray:
        """Compute a matrix product with the lower triangular linear system

        If the covariance matrix is ``K = L @ L.T`` for some lower triangular
        matrix ``L``, this method returns ``L @ y`` for some ``y``.
        """
        raise NotImplementedError

    @abstractmethod
    def condition(
        self,
        kernel: Kernel | None,
        X_train: JAXArray,
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
        *,
        include_mean: bool,
        mean_function: MeanBase,
    ) -> ConditionedComponents:
        """Build the components of a conditioned GP.

        Args:
            kernel: The kernel for the cross-covariance between observed and
                predicted data (and the predicted prior covariance), or ``None``
                when the user did not override the kernel. ``None`` means
                "predict with the training kernel"; solvers should resolve it to
                their own training kernel, and may use it as a static signal to
                enable a same-kernel fast path (this is robust under ``jax.jit``,
                where object identity is not preserved across the trace
                boundary).
            X_train: The training input coordinates.
            X_test: The coordinates of the predicted points, or ``None`` to
                predict at the training inputs.
            noise: The noise model for the predicted process.
            alpha: The precomputed :math:`K^{-1} y` for the training data.
            include_mean: If ``True``, ``mean_value`` and the returned mean
                object should include the prior ``mean_function``.
            mean_function: The prior mean function of the training GP.

        Returns:
            A :class:`ConditionedComponents` bundle.
        """
        raise NotImplementedError
