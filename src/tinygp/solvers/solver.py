from __future__ import annotations

__all__ = ["Solver", "ConditionedComponents"]

from abc import abstractmethod
from typing import Any, NamedTuple

import equinox as eqx

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise


class ConditionedComponents(NamedTuple):
    """The pieces of a conditioned process, as returned by :func:`Solver.condition`

    Everything here describes the conditional process *without* the prior mean
    function of the parent process; :class:`tinygp.GaussianProcess` adds that
    back itself.
    """

    kernel: Kernel
    """The conditional kernel; its diagonal is the conditional variance."""

    mean_value: JAXArray
    """The conditional mean evaluated at the test points."""

    solver: Solver
    """The solver for the conditioned process

    This is built by the solver that produced it, so that solver-specific
    settings (and representations) carry over. A solver that can avoid
    materializing the full ``N_test x N_test`` conditional covariance should
    return a lazy solver here (see, for example,
    :class:`tinygp.solvers.direct.LazyDirectSolver`).
    """


class Solver(eqx.Module):
    """The interface for the linear algebra backends used by a GaussianProcess

    Implementations must store the kernel and input coordinates that they were
    built with as ``kernel`` and ``X``.
    """

    kernel: Kernel
    X: JAXArray

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
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
    ) -> ConditionedComponents:
        """Build the components of the process conditioned on observed data

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data. If ``None``, the kernel used to construct this
                solver is used, and solvers can use this as a signal to enable
                specialized algorithms. (This signal is used rather than
                checking object identity because identity is not preserved
                under ``jax.jit``.)
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
            alpha: The vector ``K^{-1} @ (y - mean)`` for the observed data.

        Returns:
            A :class:`ConditionedComponents` describing the conditional
            process, excluding the prior mean function.
        """
        raise NotImplementedError
