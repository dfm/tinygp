# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["QuasisepSolver"]

from typing import Any

import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels import Kernel
from tinygp.solvers.quasisep.core import DiagQSM, LowerTriQSM, SymmQSM
from tinygp.solvers.quasisep.kernels import Quasisep
from tinygp.solvers.solver import Solver


@dataclass
class QuasisepSolver(Solver):
    X: JAXArray
    matrix: SymmQSM
    factor: LowerTriQSM

    @classmethod
    def init(
        cls, kernel: Kernel, X: JAXArray, diag: JAXArray
    ) -> "QuasisepSolver":
        assert isinstance(kernel, Quasisep)
        matrix = kernel.to_qsm(X)
        matrix += DiagQSM(d=jnp.broadcast_to(diag, matrix.diag.d.shape))
        factor = matrix.cholesky()
        return cls(X=X, matrix=matrix, factor=factor)

    def variance(self) -> JAXArray:
        return self.matrix.diag.d

    def covariance(self) -> Any:
        return self.matrix

    def normalization(self) -> JAXArray:
        return jnp.sum(jnp.log(self.factor.diag.d)) + 0.5 * self.factor.shape[
            0
        ] * np.log(2 * np.pi)

    def solve_triangular(
        self, y: JAXArray, *, transpose: bool = False
    ) -> JAXArray:
        if transpose:
            return self.factor.transpose().solve(y)
        else:
            return self.factor.solve(y)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self.factor @ y
