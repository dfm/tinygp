# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["QuasisepSolver"]

from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels import Kernel
from tinygp.kernels.quasisep import Quasisep
from tinygp.solvers.quasisep.core import DiagQSM, LowerTriQSM, SymmQSM
from tinygp.solvers.solver import Solver


@dataclass
class QuasisepSolver(Solver):
    X: JAXArray
    matrix: SymmQSM
    factor: LowerTriQSM

    @classmethod
    def init(
        cls,
        kernel: Kernel,
        X: JAXArray,
        diag: JAXArray,
        *,
        covariance: Optional[Any] = None,
    ) -> "QuasisepSolver":
        if covariance is None:
            assert isinstance(kernel, Quasisep)
            matrix = kernel.to_symm_qsm(X)
            matrix += DiagQSM(d=jnp.broadcast_to(diag, matrix.diag.d.shape))
        else:
            assert isinstance(covariance, SymmQSM)
            matrix = covariance
        factor = matrix.cholesky()
        return cls(X=X, matrix=matrix, factor=factor)

    def variance(self) -> JAXArray:
        return self.matrix.diag.d

    def covariance(self) -> JAXArray:
        return self.matrix.to_dense()

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

    @partial(jax.jit, static_argnums=(1,))
    def condition(
        self,
        kernel: Kernel,
        X_test: Optional[JAXArray],
        diag: Optional[JAXArray],
    ) -> Any:
        # We can easily compute the conditional as a QSM in the special case
        # where we are predicting at the input coordinates and a Quasisep kernel
        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(self.X)
            delta = (self.factor.inv() @ M).gram()
            if diag is not None:
                M += DiagQSM(d=jnp.broadcast_to(diag, M.diag.d.shape))
            return M - delta

        # Otherwise fall back on the slow method for now :(
        if X_test is None:
            Kss = Ks = kernel(self.X, self.X)
        else:
            Kss = kernel(X_test, X_test)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A
