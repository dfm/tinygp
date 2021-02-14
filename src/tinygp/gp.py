# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess", "zero_mean", "constant_mean"]

from typing import Union, Optional, Callable

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from .kernels import Kernel

LOG2PI = jnp.log(2 * jnp.pi)

Mean = Callable[[jnp.ndarray], jnp.ndarray]


class GaussianProcess:
    def __init__(
        self,
        kernel: Kernel,
        X: jnp.ndarray,
        *,
        diag: Union[jnp.ndarray, float] = 0.0,
        mean: Optional[Union[Mean, jnp.ndarray]] = None,
        lower: bool = True,
    ):
        # Format input
        self.X = _pad_input(X)
        self.diag = jnp.asarray(diag)

        # Parse the mean function
        if callable(mean):
            self.mean = mean
        else:
            if mean is None:
                self.mean = zero_mean
            else:
                self.mean = constant_mean(jnp.asarray(mean))

        # Evaluate the covariance matrix and factorize the matrix
        self.kernel = kernel
        self.K0 = self.kernel.evaluate(X, X)
        self.K = jax.ops.index_add(
            self.K0, jnp.diag_indices(self.X.shape[0]), self.diag
        )
        self.lower = lower
        self.chol = linalg.cholesky(self.K, lower=self.lower)
        self.norm = (
            jnp.sum(jnp.log(jnp.diag(self.chol)))
            + 0.5 * self.X.shape[0] * LOG2PI
        )

    def condition(self, y: jnp.ndarray) -> jnp.ndarray:
        self.y = jnp.asarray(y)
        self.resid = self.y - self.mean(self.X)
        self.alpha = linalg.solve_triangular(
            self.chol, self.resid, lower=self.lower
        )
        return -0.5 * jnp.sum(jnp.square(self.alpha)) - self.norm

    def predict(
        self,
        X_test: jnp.ndarray = None,
        *,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ):
        if not hasattr(self, "y"):
            raise RuntimeError("'condition' must be called first")

        # Compute the conditional
        if X_test is None:
            X_test = self.X
            K_testT = linalg.solve_triangular(
                self.chol, self.K0, lower=self.lower
            )
            delta = (
                linalg.solve_triangular(self.chol, self.diag, lower=self.lower)
                * self.alpha
            )
            if include_mean:
                mu = self.y - delta
            else:
                mu = self.resid - delta
        else:
            X_test = _pad_input(X_test)
            K_testT = linalg.solve_triangular(
                self.chol,
                self.kernel.evaluate(self.X, X_test),
                lower=self.lower,
            )
            mu = K_testT.T @ self.alpha
            if include_mean:
                mu += self.mean(X_test)

        if not (return_var or return_cov):
            return mu

        if return_var:
            var = self.kernel.evaluate_diag(X_test) - jnp.sum(
                jnp.square(K_testT), axis=0
            )
            return mu, var

        cov = self.kernel.evaluate(X_test, X_test) - K_testT.T @ K_testT
        return mu, cov


def zero_mean(X: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(X.shape[0])


def constant_mean(value: jnp.ndarray) -> Mean:
    _value = value

    def mean(X: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(X.shape[0], _value)

    return mean


def _pad_input(X: jnp.ndarray) -> jnp.ndarray:
    X = jnp.atleast_1d(X)
    if X.ndim == 1:
        X = X[:, None]
    return X
