# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["GaussianProcess"]

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from .kernels import Kernel
from .means import Mean, constant_mean, zero_mean
from .types import JAXArray


class GaussianProcess:
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (JAXArray): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``,
            the size of the data set.
        diag (JAXArray, optional): The value to add to the diagonal of the
            covariance matrix, often used to capture measurement uncertainty.
            This should be a scalar or have the shape ``(N_data,)``.
        mean: (Mean, optional): A callable or constant mean function that will
            be evaluated with the ``X`` as input: ``mean(X)``
    """

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        *,
        diag: Union[JAXArray, float] = 0.0,
        mean: Optional[Union[Mean, JAXArray]] = None,
    ):
        self.X = X
        self.diag = diag

        # Parse the mean function
        if callable(mean):
            self.mean_function = mean
        else:
            if mean is None:
                self.mean_function = zero_mean
            else:
                self.mean_function = constant_mean(mean)
        self.mean_function = jax.vmap(self.mean_function)
        self.loc = self.mean = self.mean_function(self.X)
        assert self.mean.ndim == 1

        # Evaluate the covariance matrix and factorize the matrix
        self.kernel = kernel
        self.base_covariance_matrix = self.kernel(X, X)
        self.dtype = self.base_covariance_matrix.dtype
        self.num_data = self.base_covariance_matrix.shape[0]
        self.covariance_matrix = self.base_covariance_matrix.at[  # type: ignore
            jnp.diag_indices(self.num_data)
        ].add(
            self.diag
        )
        self.scale_tril = linalg.cholesky(self.covariance_matrix, lower=True)
        self.norm = jnp.sum(jnp.log(jnp.diag(self.scale_tril)))
        self.norm += 0.5 * self.num_data * jnp.log(2 * jnp.pi)

    def condition(self, y: JAXArray) -> JAXArray:
        """Condition the process on observed data

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.

        Returns:
            The marginal likelihood of this model, evaluated at ``y``.
        """
        return self._condition(self._get_alpha(y))

    def predict(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        kernel: Optional[Kernel] = None,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ) -> Union[JAXArray, Tuple[JAXArray, JAXArray]]:
        """Predict the GP model at new test points conditioned on observed data

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.
            X_test (JAXArray, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            return_var (bool, optional): If ``True``, the variance of the
                predicted values at ``X_test`` will be returned.
            return_cov (bool, optional): If ``True``, the covariance of the
                predicted values at ``X_test`` will be returned. If
                ``return_var`` is ``True``, this flag will be ignored.

        Returns:
            The mean of the predictive model evaluated at ``X_test``, with shape
            ``(N_test,)`` where ``N_test`` is the zeroth dimension of
            ``X_test``. If either ``return_var`` or ``return_cov`` is ``True``,
            the variance or covariance of the predicted process will also be
            returned with shape ``(N_test,)`` or ``(N_test, N_test)``
            respectively.
        """
        alpha = self._get_alpha(y)
        return self._predict(
            y, alpha, X_test, kernel, include_mean, return_var, return_cov
        )

    def condition_and_predict(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        kernel: Optional[Kernel] = None,
        include_mean: bool = True,
        return_var: bool = False,
        return_cov: bool = False,
    ) -> Tuple[JAXArray, Union[JAXArray, Tuple[JAXArray, JAXArray]]]:
        """Condition on observed data and return the predictive process

        This combines :func:`GaussianProcess.condition` and
        :func:`GaussianProcess.predict` into a single operation which will be
        somewhat more efficient than calling them both separately. See those
        docstrings for a description of all the arguments.
        """
        alpha = self._get_alpha(y)
        return self._condition(alpha), self._predict(
            y, alpha, X_test, kernel, include_mean, return_var, return_cov
        )

    def sample(
        self,
        key: jax.random.KeyArray,
        shape: Optional[Sequence[int]] = None,
    ) -> JAXArray:
        """Generate samples from the prior process

        Args:
            key: A ``jax`` random number key array. shape (tuple, optional): The
            number and shape of samples to
                generate.

        Returns:
            The sampled realizations from the process with shape ``(N_data,) +
            shape`` where ``N_data`` is the zeroth dimension of the ``X``
            coordinates provided when instantiating this process.
        """
        return self._sample(key, shape)

    def numpyro_dist(self, **kwargs):  # type: ignore
        """Get the numpyro MultivariateNormal distribution for this process"""
        import numpyro.distributions as dist

        return dist.MultivariateNormal(
            loc=self.loc, scale_tril=self.scale_tril, **kwargs
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _sample(
        self,
        key: jax.random.KeyArray,
        shape: Optional[Sequence[int]],
    ) -> JAXArray:
        if shape is None:
            shape = (self.num_data,)
        else:
            shape = (self.num_data,) + tuple(shape)
        normal_samples = jax.random.normal(key, shape=shape, dtype=self.dtype)
        return self.mean + jnp.einsum(
            "...ij,...j->...i", self.scale_tril, normal_samples
        )

    def _condition(self, alpha: JAXArray) -> JAXArray:
        return -0.5 * jnp.sum(jnp.square(alpha)) - self.norm

    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return linalg.solve_triangular(
            self.scale_tril, y - self.loc, lower=True
        )

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
    def _predict(
        self,
        y: JAXArray,
        alpha: JAXArray,
        X_test: Optional[JAXArray],
        kernel: Optional[Kernel],
        include_mean: bool,
        return_var: bool,
        return_cov: bool,
    ) -> Union[JAXArray, Tuple[JAXArray, JAXArray]]:
        if X_test is None and kernel is None:
            kernel = self.kernel
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
            if X_test is None:
                X_test = self.X
            if kernel is None:
                kernel = self.kernel

            K_testT = linalg.solve_triangular(
                self.scale_tril,
                kernel(self.X, X_test),
                lower=True,
            )
            mu = K_testT.T @ alpha
            if include_mean:
                mu += self.mean_function(X_test)

            if not (return_var or return_cov):
                return mu

        if return_var:
            var = kernel(X_test) - jnp.sum(jnp.square(K_testT), axis=0)
            return mu, var

        cov = kernel(X_test, X_test) - K_testT.T @ K_testT
        return mu, cov
