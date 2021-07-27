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


class GaussianProcess:
    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        *,
        diag: Union[JAXArray, float] = 0.0,
        mean: Optional[Union[Mean, JAXArray]] = None,
    ):
        # Format input
        self.X = _pad_input(X)
        assert self.X.ndim == 2
        self.size = self.X.shape[0]
        self.diag = jnp.broadcast_to(diag, (self.size,))

        # Parse the mean function
        if callable(mean):
            self.mean_function = mean
        else:
            if mean is None:
                self.mean_function = zero_mean
            else:
                self.mean_function = constant_mean(jnp.asarray(mean))
        self.loc = self.mean = self.mean_function(self.X)
        assert self.mean.ndim == 1

        # Evaluate the covariance matrix and factorize the matrix
        self.kernel = kernel
        self.base_covariance_matrix = self.kernel(X, X)
        self.covariance_matrix = jax.ops.index_add(
            self.base_covariance_matrix,
            jnp.diag_indices(self.X.shape[0]),
            self.diag,
        )
        self.scale_tril = linalg.cholesky(self.covariance_matrix, lower=True)
        self.norm = jnp.sum(jnp.log(jnp.diag(self.scale_tril)))
        self.norm += 0.5 * self.X.shape[0] * jnp.log(2 * jnp.pi)

    def condition(self, y: JAXArray) -> JAXArray:
        y = jnp.broadcast_to(y, (self.size,))
        return self._condition(self._get_alpha(y))

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
        return self._predict(
            y,
            alpha,
            X_test=X_test,
            include_mean=include_mean,
            return_var=return_var,
            return_cov=return_cov,
        )

    def condition_and_predict(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ) -> Tuple[JAXArray, Union[JAXArray, Tuple[JAXArray, JAXArray]]]:
        y = jnp.broadcast_to(y, (self.size,))
        alpha = self._get_alpha(y)
        return self._condition(alpha), self._predict(
            y,
            alpha,
            X_test=X_test,
            include_mean=include_mean,
            return_var=return_var,
            return_cov=return_cov,
        )

    def _condition(self, alpha: JAXArray) -> JAXArray:
        return -0.5 * jnp.sum(jnp.square(alpha)) - self.norm

    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return linalg.solve_triangular(
            self.scale_tril, y - self.loc, lower=True
        )

    def _predict(
        self,
        y: JAXArray,
        alpha: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ) -> Union[JAXArray, Tuple[JAXArray, JAXArray]]:

        # Compute the conditional
        if X_test is None:
            delta = self.diag * linalg.solve_triangular(
                self.scale_tril, alpha, lower=True, trans=1
            )
            mu = y - delta
            if not include_mean:
                mu -= self.loc

            if not (return_var or return_cov):
                return mu

            X_test = self.X
            K_testT = linalg.solve_triangular(
                self.scale_tril, self.base_covariance_matrix, lower=True
            )

        else:
            X_test = _pad_input(X_test)
            K_testT = linalg.solve_triangular(
                self.scale_tril,
                self.kernel(self.X, X_test),
                lower=True,
            )
            mu = K_testT.T @ alpha
            if include_mean:
                mu += self.mean_function(X_test)

            if not (return_var or return_cov):
                return mu

        if return_var:
            var = self.kernel.evaluate_diag(X_test) - jnp.sum(
                jnp.square(K_testT), axis=0
            )
            return mu, var

        cov = self.kernel(X_test, X_test) - K_testT.T @ K_testT
        return mu, cov


def _pad_input(X: JAXArray) -> JAXArray:
    X = jnp.atleast_1d(X)
    if X.ndim == 1:
        X = X[:, None]
    return X
