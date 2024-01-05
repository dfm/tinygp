from __future__ import annotations

__all__ = ["DirectSolver"]

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

from tinygp import kernels
from tinygp.helpers import JAXArray
from tinygp.noise import Noise
from tinygp.solvers.solver import Solver


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

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ):
        """Build a :class:`DirectSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed array with the covariance
                matrix. This should be equal to the result of calling ``kernel``
                and adding ``diag``, but that is not checked.
        """
        self.X = X
        self.variance_value = kernel(X) + noise.diagonal()
        if covariance is None:
            covariance = kernel(X, X) + noise
        self.covariance_value = covariance
        self.scale_tril = linalg.cholesky(covariance, lower=True)

    def variance(self) -> JAXArray:
        return self.variance_value

    def covariance(self) -> JAXArray:
        return self.covariance_value

    def normalization(self) -> JAXArray:
        return jnp.sum(
            jnp.log(jnp.diag(self.scale_tril))
        ) + 0.5 * self.scale_tril.shape[0] * np.log(2 * np.pi)

    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        if transpose:
            return linalg.solve_triangular(self.scale_tril, y, lower=True, trans=1)
        else:
            return linalg.solve_triangular(self.scale_tril, y, lower=True)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return jnp.einsum("ij,j...->i...", self.scale_tril, y)

    def condition(
        self, kernel: kernels.Kernel, X_test: JAXArray | None, noise: Noise
    ) -> Any:
        """Compute the covariance matrix for a conditional GP

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
        """
        if X_test is None:
            Ks = kernel(self.X, self.X)
            Kss = Ks + noise
        else:
            Ks = kernel(self.X, X_test)
            Kss = kernel(X_test, X_test) + noise

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A
