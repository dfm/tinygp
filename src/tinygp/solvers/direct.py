# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["DirectSolver"]

from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

from tinygp import kernels
from tinygp.helpers import JAXArray, dataclass
from tinygp.solvers.solver import Solver


@dataclass
class DirectSolver(Solver):
    """A direct solver that uses ``jax``'s built in Cholesky factorization

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`DirectSolver.init` method instead of the
    usual constructor.
    """

    X: JAXArray
    variance_value: JAXArray
    covariance_value: JAXArray
    scale_tril: JAXArray

    @classmethod
    def init(
        cls,
        kernel: kernels.Kernel,
        X: JAXArray,
        diag: JAXArray,
        *,
        covariance: Optional[Any] = None,
    ) -> "DirectSolver":
        """Build a :class:`DirectSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            diag: An extra diagonal component to add to the covariance matrix.
            covariance: Optionally, a pre-computed array with the covariance
                matrix. This should be equal to the result of calling ``kernel``
                and adding ``diag``, but that is not checked.
        """
        variance = kernel(X) + diag
        if covariance is None:
            covariance = construct_covariance(kernel, X, diag)
        scale_tril = linalg.cholesky(covariance, lower=True)
        return cls(
            X=X,
            variance_value=variance,
            covariance_value=covariance,
            scale_tril=scale_tril,
        )

    def variance(self) -> JAXArray:
        return self.variance_value

    def covariance(self) -> JAXArray:
        return self.covariance_value

    def normalization(self) -> JAXArray:
        return jnp.sum(
            jnp.log(jnp.diag(self.scale_tril))
        ) + 0.5 * self.scale_tril.shape[0] * np.log(2 * np.pi)

    def solve_triangular(
        self, y: JAXArray, *, transpose: bool = False
    ) -> JAXArray:
        if transpose:
            return linalg.solve_triangular(
                self.scale_tril, y, lower=True, trans=1
            )
        else:
            return linalg.solve_triangular(self.scale_tril, y, lower=True)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return jnp.einsum("ij,j...->i...", self.scale_tril, y)

    def condition(
        self,
        kernel: kernels.Kernel,
        X_test: Optional[JAXArray],
        diag: Optional[JAXArray],
    ) -> Any:
        """Compute the covariance matrix for a conditional GP

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            diag: Any extra variance to add to the diagonal of the predicted
                model.
        """
        if X_test is None:
            Ks = kernel(self.X, self.X)
            if diag is None:
                Kss = Ks
            else:
                Kss = construct_covariance(kernel, self.X, diag)
        else:
            if diag is None:
                Kss = kernel(X_test, X_test)
            else:
                Kss = construct_covariance(kernel, X_test, diag)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A


def construct_covariance(
    kernel: kernels.Kernel, X: JAXArray, diag: JAXArray
) -> JAXArray:
    covariance = kernel(X, X)
    covariance = covariance.at[jnp.diag_indices(covariance.shape[0])].add(diag)  # type: ignore
    return covariance
