from __future__ import annotations

__all__ = ["DirectSolver", "dense_condition"]

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

from tinygp import kernels
from tinygp.helpers import JAXArray
from tinygp.noise import Noise
from tinygp.solvers.solver import ConditionedComponents, Solver


class DirectSolver(Solver):
    """A direct solver that uses ``jax``'s built in Cholesky factorization

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`DirectSolver.init` method instead of the
    usual constructor.
    """

    kernel: kernels.Kernel
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
        variance: JAXArray | None = None,
    ):
        """Build a :class:`DirectSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed array with the covariance
                matrix. This should be equal to the result of calling ``kernel``
                and adding ``diag``, but that is not checked.
            variance: Optionally, a pre-computed array with the diagonal of
                ``covariance``. If not provided, this is evaluated using
                ``kernel`` rather than read off ``covariance``, so that (under
                ``jax.jit``) a caller that only needs the variance never forces
                the full matrix to be built.
        """
        self.kernel = kernel
        self.X = X
        if variance is None:
            variance = kernel(X) + noise.diagonal()
        self.variance_value = variance
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
        self,
        kernel: kernels.Kernel | None,
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
    ) -> ConditionedComponents:
        return dense_condition(self, kernel, X_test, noise, alpha)


def dense_condition(
    solver: Solver,
    kernel: kernels.Kernel | None,
    X_test: JAXArray | None,
    noise: Noise,
    alpha: JAXArray,
) -> ConditionedComponents:
    """The generic implementation of :func:`Solver.condition`

    This computes the conditional covariance as a dense matrix, and it can be
    used by any solver since it only requires ``solve_triangular`` and the
    ``X`` and ``kernel`` attributes. The cross covariance block ``Ks`` is
    evaluated once and shared by the mean, the variance, and the covariance.
    The variance is passed to the resulting :class:`DirectSolver` explicitly
    so that, under ``jax.jit``, a caller that only reads the conditional mean
    and variance never materializes or factorizes the ``N_test x N_test``
    covariance: that computation is dead code and XLA eliminates it.
    """
    X_train = solver.X
    kernel = solver.kernel if kernel is None else kernel
    Xt = X_train if X_test is None else X_test
    Ks = kernel(X_train, Xt)
    A = solver.solve_triangular(Ks)
    var = kernel(Xt) - jnp.sum(jnp.square(A), axis=0) + noise.diagonal()
    Kss = (Ks if X_test is None else kernel(Xt, Xt)) - A.transpose() @ A + noise
    cond_kernel = kernels.Conditioned(X_train, solver, kernel)
    return ConditionedComponents(
        kernel=cond_kernel,
        mean_value=jnp.dot(alpha, Ks),
        solver=DirectSolver(cond_kernel, Xt, noise, covariance=Kss, variance=var),
    )
