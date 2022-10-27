# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["QuasisepSolver"]

from functools import wraps
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise
from tinygp.solvers.quasisep.core import LowerTriQSM, SymmQSM
from tinygp.solvers.solver import Solver


def handle_sorting(func: Callable[..., JAXArray]) -> Callable[..., JAXArray]:
    @wraps(func)
    def wrapped(
        self: "QuasisepSolver", y: JAXArray, *args: Any, **kwargs: Any
    ) -> JAXArray:
        if self.inds_to_sorted is not None:
            y = y[self.inds_to_sorted]
        r = func(self, y, *args, **kwargs)
        if self.sorted_to_inds is not None:
            return r[self.sorted_to_inds]
        return r

    return wrapped


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
    inds_to_sorted: Optional[JAXArray]
    sorted_to_inds: Optional[JAXArray]

    @classmethod
    def init(
        cls,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Optional[Any] = None,
        sort: bool = True,
    ) -> "QuasisepSolver":
        """Build a :class:`QuasisepSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed
                :class:`tinygp.solvers.quasisep.core.QSM` with the covariance
                matrix.
        """
        from tinygp.kernels.quasisep import Quasisep

        inds_to_sorted = None
        sorted_to_inds = None
        if covariance is None:
            assert isinstance(kernel, Quasisep)

            if sort:
                inds_to_sorted = jnp.argsort(kernel.coord_to_sortable(X))
                sorted_to_inds = (
                    jnp.empty_like(inds_to_sorted)
                    .at[inds_to_sorted]
                    .set(jnp.arange(len(inds_to_sorted)))
                )
                X = X[inds_to_sorted]

            matrix = kernel.to_symm_qsm(X)
            matrix += noise.to_qsm()

        else:
            assert isinstance(covariance, SymmQSM)
            matrix = covariance

        factor = matrix.cholesky()
        return cls(
            X=X,
            matrix=matrix,
            factor=factor,
            inds_to_sorted=inds_to_sorted,
            sorted_to_inds=sorted_to_inds,
        )

    def variance(self) -> JAXArray:
        return self.matrix.diag.d
        if self.sorted_to_inds is None:
            return self.matrix.diag.d
        return self.matrix.diag.d[self.sorted_to_inds]

    def covariance(self) -> JAXArray:
        cov = self.matrix.to_dense()
        return cov
        if self.sorted_to_inds is None:
            return cov
        return cov[self.sorted_to_inds[:, None], self.sorted_to_inds[None, :]]

    def normalization(self) -> JAXArray:
        return jnp.sum(jnp.log(self.factor.diag.d)) + 0.5 * self.factor.shape[
            0
        ] * np.log(2 * np.pi)

    @handle_sorting
    def solve_triangular(
        self, y: JAXArray, *, transpose: bool = False
    ) -> JAXArray:
        if transpose:
            return self.factor.transpose().solve(y)
        else:
            return self.factor.solve(y)

    @handle_sorting
    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self.factor @ y

    def condition(
        self, kernel: Kernel, X_test: Optional[JAXArray], noise: Noise
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
            noise: The noise model for the predicted process.
        """
        from tinygp.kernels.quasisep import Quasisep

        # We can easily compute the conditional as a QSM in the special case
        # where we are predicting at the input coordinates and a Quasisep kernel
        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(self.X)
            delta = (self.factor.inv() @ M).gram()
            M += noise.to_qsm()
            return M - delta

        # Otherwise fall back on the slow method for now :(
        if X_test is None:
            Kss = Ks = kernel(self.X, self.X)
        else:
            Kss = kernel(X_test, X_test)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A
