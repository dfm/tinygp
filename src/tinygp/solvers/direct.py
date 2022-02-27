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
        if covariance is None:
            variance = kernel(X) + diag
            covariance = construct_covariance(kernel, X, diag)
        else:
            variance = jnp.diag(covariance)
        scale_tril = linalg.cholesky(covariance, lower=True)
        return cls(
            X=X,
            variance_value=variance,
            covariance_value=covariance,
            scale_tril=scale_tril,
        )

    def variance(self) -> JAXArray:
        return self.variance_value

    def covariance(self) -> Any:
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


def construct_covariance(
    kernel: kernels.Kernel, X: JAXArray, diag: JAXArray
) -> JAXArray:
    covariance = kernel(X, X)
    covariance = covariance.at[jnp.diag_indices(covariance.shape[0])].add(diag)  # type: ignore
    return covariance
