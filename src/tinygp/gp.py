# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["GaussianProcess"]

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from .kernels import Kernel
from .means import Mean, constant_mean, zero_mean
from .types import JAXArray

try:
    from numpyro.distributions import MultivariateNormal
except ImportError:
    MultivariateNormal = None


class GaussianProcess:
    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        *,
        diag: Union[JAXArray, float] = 0.0,
        mean: Optional[Union[Mean, JAXArray]] = None,
        lower: bool = True,
    ):
        # Format input
        self.X = _pad_input(X)
        assert self.X.ndim == 2
        self.size = self.X.shape[0]
        self.diag = jnp.broadcast_to(diag, (self.size,))

        # Parse the mean function
        if callable(mean):
            self.mean = mean
        else:
            if mean is None:
                self.mean = zero_mean
            else:
                self.mean = constant_mean(jnp.asarray(mean))
        self.mean_value = self.mean(self.X)
        assert self.mean_value.ndim == 1

        # Evaluate the covariance matrix and factorize the matrix
        self.kernel = kernel
        self.K0 = self.kernel.evaluate(X, X)
        self.K = jax.ops.index_add(
            self.K0, jnp.diag_indices(self.X.shape[0]), self.diag
        )
        self.lower = lower
        self.chol = linalg.cholesky(self.K, lower=self.lower)
        self.norm = jnp.sum(jnp.log(jnp.diag(self.chol)))
        self.norm += 0.5 * self.X.shape[0] * jnp.log(2 * jnp.pi)

    def condition(self, y: JAXArray) -> JAXArray:
        y = jnp.broadcast_to(y, (self.size,))
        return -0.5 * jnp.sum(jnp.square(self._get_alpha(y))) - self.norm

    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return linalg.solve_triangular(
            self.chol,
            y - self.mean_value,
            lower=self.lower,
        )

    def predict(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ) -> Union[JAXArray, Tuple[JAXArray, JAXArray]]:
        y = jnp.broadcast_to(y, (self.size,))
        alpha = self._get_alpha(y)

        # Compute the conditional
        if X_test is None:
            delta = self.diag * linalg.solve_triangular(
                self.chol, alpha, lower=self.lower, trans=1
            )
            mu = y - delta
            if not include_mean:
                mu -= self.mean_value

            if not (return_var or return_cov):
                return mu

            X_test = self.X
            K_testT = linalg.solve_triangular(
                self.chol, self.K0, lower=self.lower
            )

        else:
            X_test = _pad_input(X_test)
            K_testT = linalg.solve_triangular(
                self.chol,
                self.kernel.evaluate(self.X, X_test),
                lower=self.lower,
            )
            mu = K_testT.T @ alpha
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

    def numpyro_marginal(self) -> MultivariateNormal:
        if MultivariateNormal is None:
            raise ImportError("numpyro must be installed")

        if self.lower:
            scale_tril = self.chol
        else:
            scale_tril = self.chol.T

        return MultivariateNormal(loc=self.mean_value, scale_tril=scale_tril)

    def numpyro_conditional(
        self, y: JAXArray, X_test: Optional[JAXArray] = None
    ) -> MultivariateNormal:
        if MultivariateNormal is None:
            raise ImportError("numpyro must be installed")

        mu, cov = self.predict(
            y, X_test, include_mean=True, return_var=False, return_cov=True
        )
        return MultivariateNormal(loc=mu, covariance_matrix=cov)


def _pad_input(X: JAXArray) -> JAXArray:
    X = jnp.atleast_1d(X)
    if X.ndim == 1:
        X = X[:, None]
    return X
