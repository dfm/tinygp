from __future__ import annotations

__all__ = ["QuasisepSolver"]

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise
from tinygp.solvers.quasisep.core import LowerTriQSM, SymmQSM
from tinygp.solvers.solver import Solver


def _sorted_method(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(
        self: QuasisepSolver, y: JAXArray, *args: Any, **kwargs: Any
    ) -> JAXArray:
        if self.argsort_inds is not None:
            y = y[self.argsort_inds]

        r = func(self, y, *args, **kwargs)

        if self.unsort_inds is not None:
            return r[self.unsort_inds]
        else:
            return r

    return wrapped


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
    argsort_inds: JAXArray | None
    unsort_inds: JAXArray | None

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
        assume_sorted: bool = False,
        check_sorted: bool | None = None,
    ):
        """Build a :class:`QuasisepSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed
                :class:`tinygp.solvers.quasisep.core.QSM` with the covariance
                matrix.
            assume_sorted: If ``True``, assume that the input coordinates are
                sorted. If ``False``, check that they are sorted and throw an
                error if they are not. This can introduce a runtime overhead,
                and you can pass ``assume_sorted=True`` to get the best
                performance.
        """
        if TYPE_CHECKING:
            from tinygp.kernels.quasisep import Quasisep

            assert isinstance(kernel, Quasisep)

        if covariance is None:
            if assume_sorted or check_sorted:
                self.argsort_inds = None
                self.unsort_inds = None

                if check_sorted:
                    jax.debug.callback(_check_sorted, kernel.coord_to_sortable(X))

            else:
                self.argsort_inds = jnp.argsort(kernel.coord_to_sortable(X))
                self.unsort_inds = jnp.argsort(self.argsort_inds)

            matrix = kernel.to_symm_qsm(_apply_sort(X, self.argsort_inds))
            matrix += noise.to_qsm(argsort_inds=self.argsort_inds)

        else:
            if TYPE_CHECKING:
                assert isinstance(covariance, SymmQSM)
            self.argsort_inds = None
            self.unsort_inds = None
            matrix = covariance

        self.X = X
        self.matrix = matrix
        self.factor = matrix.cholesky()

    def variance(self) -> JAXArray:
        if self.unsort_inds is None:
            return self.matrix.diag.d
        else:
            return self.matrix.diag.d[self.unsort_inds]

    def covariance(self) -> JAXArray:
        if self.unsort_inds is None:
            return self.matrix.to_dense()
        else:
            return self.matrix.to_dense()[
                self.unsort_inds[:, None], self.unsort_inds[None, :]
            ]

    def normalization(self) -> JAXArray:
        return jnp.sum(jnp.log(self.factor.diag.d)) + 0.5 * self.factor.shape[
            0
        ] * np.log(2 * np.pi)

    @_sorted_method
    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        if transpose:
            return self.factor.transpose().solve(y)
        else:
            return self.factor.solve(y)

    @_sorted_method
    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self.factor @ y

    def condition(self, kernel: Kernel, X_test: JAXArray | None, noise: Noise) -> Any:
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
            M = kernel.to_symm_qsm(_apply_sort(self.X, self.argsort_inds))
            delta = (self.factor.inv() @ M).gram()
            M += noise.to_qsm(argsort_inds=self.argsort_inds)
            return M - delta

        # Otherwise fall back on the slow method for now :(
        if X_test is None:
            Kss = Ks = kernel(self.X, self.X)
        else:
            Kss = kernel(X_test, X_test)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A


def _apply_sort(X: JAXArray, inds: JAXArray | None) -> JAXArray:
    if inds is None:
        return X
    else:
        return jax.tree_util.tree_map(lambda v: v[inds], X)


def _check_sorted(X: JAXArray) -> None:
    if np.any(np.diff(X) < 0.0):
        raise ValueError(
            "Input coordinates must be sorted in order to use the QuasisepSolver"
        )
