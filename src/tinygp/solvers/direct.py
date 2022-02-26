# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["DirectSolver"]

import numpy as np
import jax.numpy as jnp
from jax.scipy import linalg

from tinygp.solvers.solver import Solver
from tinygp.kernels import Kernel
from tinygp.helpers import JAXArray, dataclass


@dataclass
class DirectSolver(Solver):
    kernel: Kernel
    X: JAXArray
    diag: JAXArray
    scale_tril: JAXArray

    @classmethod
    def init(
        cls, kernel: Kernel, X: JAXArray, diag: JAXArray
    ) -> "DirectSolver":
        diag = jnp.zeros(()) if diag is None else diag
        covariance = construct_covariance(kernel, X, diag)
        scale_tril = linalg.cholesky(covariance, lower=True)
        return cls(
            kernel=kernel,
            X=X,
            diag=diag,
            scale_tril=scale_tril,
        )

    def variance(self) -> JAXArray:
        return self.kernel(self.X)

    def covariance(self) -> JAXArray:
        return construct_covariance(self.kernel, self.X, self.diag)

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


def construct_covariance(
    kernel: Kernel, X: JAXArray, diag: JAXArray
) -> JAXArray:
    covariance = kernel(X, X)
    covariance = covariance.at[jnp.diag_indices(covariance.shape[0])].add(diag)  # type: ignore
    return covariance
