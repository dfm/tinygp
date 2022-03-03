# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["QuasisepSolver"]

from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.core import DiagQSM, LowerTriQSM, SymmQSM
from tinygp.solvers.solver import Solver


@dataclass
class QuasisepSolver(Solver):
    """A scalable solver that uses quasiseparable matrices

    Take a look at the documentation for the :ref:`api-solvers-quasisep`, for
    more technical details.

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`QuasisepSolver.init` method instead of the
    usual constructor.
    """

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
        """Build a :class:`QuasisepSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            diag: An extra diagonal component to add to the covariance matrix.
            covariance: Optionally, a pre-computed
                :class:`tinygp.solvers.quasisep.core.QSM` with the covariance
                matrix.
        """
        from tinygp.kernels.quasisep import Quasisep

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
        """Compute the covariance matrix for a conditional GP

        In the case where the prediction is made at the input coordinates with a
        :class:`tinygp.kernels.quasisep.Quasisep` kernel, this will return the
        quasiseparable representation of the conditional matrix. Otherwise, it
        will use scalable methods where possible, but return a dense
        representation of the covariance, so be careful when predicting at a
        large number of test points!

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            diag: Any extra variance to add to the diagonal of the predicted
                model.
        """
        from tinygp.kernels.quasisep import Quasisep

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
