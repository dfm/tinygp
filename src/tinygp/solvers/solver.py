from __future__ import annotations

__all__ = ["Solver"]

from abc import abstractmethod
from typing import Any

import equinox as eqx

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise


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
    def condition(self, kernel: Kernel, X_test: JAXArray | None, noise: Noise) -> Any:
        raise NotImplementedError
