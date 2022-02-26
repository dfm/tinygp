# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["GaussianProcess"]

from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from tinygp import kernels, means
from tinygp.helpers import JAXArray


class GaussianProcess:
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function X (JAXArray): The input
            coordinates. This can be any PyTree that is compatible with
            ``kernel`` where the zeroth dimension is ``N_data``, the size of the
            data set.
        diag (JAXArray, optional): The value to add to the diagonal of the
            covariance matrix, often used to capture measurement uncertainty.
            This should be a scalar or have the shape ``(N_data,)``.
        mean (Callable, optional): A callable or constant mean function that
            will be evaluated with the ``X`` as input: ``mean(X)``
        mean_value (JAXArray, optional): The mean precomputed at the location
            of the data.
    """

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        *,
        diag: Optional[JAXArray] = None,
        mean: Optional[Union[Callable[[JAXArray], JAXArray], JAXArray]] = None,
        mean_value: Optional[JAXArray] = None,
    ):
        self.X = X
        self.diag = jnp.zeros(()) if diag is None else diag

        # Parse the mean function
        if callable(mean):
            self.mean_function = mean
        else:
            if mean is None:
                self.mean_function = means.Mean(jnp.zeros(()))
            else:
                self.mean_function = means.Mean(mean)
        if mean_value is None:
            mean_value = jax.vmap(self.mean_function)(self.X)
        self.loc = self.mean = mean_value
        if self.mean.ndim != 1:
            raise ValueError(
                "Invalid mean shape: "
                f"expected ndim = 1, got ndim={self.mean.ndim}"
            )

        # Evaluate the variance of the process
        self.kernel = kernel
        self.variance = self.kernel(X) + self.diag
        self.num_data = self.variance.shape[0]
        self.dtype = self.variance.dtype

        # Evaluate the covariance matrix
        self.base_covariance = self.kernel(X, X)
        self.covariance = self.base_covariance.at[
            jnp.diag_indices(self.num_data)
        ].add(self.diag)

        # Factorize the matrix and compute the log prob normalization
        self.scale_tril = linalg.cholesky(self.covariance, lower=True)
        self.norm = jnp.sum(jnp.log(jnp.diag(self.scale_tril)))
        self.norm += 0.5 * self.num_data * jnp.log(2 * jnp.pi)

    def log_probability(self, y: JAXArray) -> JAXArray:
        """Compute the log probability of this multivariate normal

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.

        Returns:
            The marginal log probability of this multivariate normal model,
            evaluated at ``y``.
        """
        return self._compute_log_prob(self._get_alpha(y))

    def condition(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        diag: Optional[JAXArray] = None,
        include_mean: bool = True,
        kernel: Optional[kernels.Kernel] = None,
    ) -> ConditionResult:
        """Condition the model on observed data and

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.
            X_test (JAXArray, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.
            diag (JAXArray, optional): Will be passed as the diagonal to the
                conditioned ``GaussianProcess`` object, so this can be used to
                introduce, for example, observational noise to predicted data.
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            kernel (Kernel, optional): A kernel to optionally specify the
                covariance between the observed data and predicted data. See
                :ref:`mixture` for an example.

        Returns:
            A named tuple where the first element ``log_probability`` is the log
            marginal probability of the model, and the second element ``gp`` is
            the :class:`GaussianProcess` object describing the conditional
            distribution evaluated at ``X_test``.
        """

        alpha = self._get_alpha(y)
        log_prob = self._compute_log_prob(alpha)

        mean_value = None
        if X_test is None:
            X_test = self.X

            # In this common case (where we're predicting the GP at the data
            # points, using the original kernel), the mean is especially fast to
            # compute; so let's use that calculation here.
            if kernel is None:
                delta = self.diag * linalg.solve_triangular(
                    self.scale_tril, alpha, lower=True, trans=1
                )
                mean_value = y - delta
                if not include_mean:
                    mean_value -= self.loc

        if kernel is None:
            kernel = self.kernel

        # The conditional GP will also be a GP with the mean an covariance
        # specified by a :class:`tinygp.means.Conditioned` and
        # :class:`tinygp.kernels.Conditioned` respectively.
        gp = GaussianProcess(
            kernels.Conditioned(self.X, self.scale_tril, kernel),
            X_test,
            diag=diag,
            mean=means.Conditioned(
                self.X,
                alpha,
                self.scale_tril,
                kernel,
                include_mean=include_mean,
                mean_function=self.mean_function,  # type: ignore
            ),
            mean_value=mean_value,
        )

        return ConditionResult(log_prob, gp)

    def predict(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray] = None,
        *,
        kernel: Optional[kernels.Kernel] = None,
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
        import warnings

        warnings.warn(
            "The 'predict' method is deprecated and 'condition' should be preferred",
            DeprecationWarning,
            stacklevel=2,
        )

        _, cond = self.condition(
            y, X_test, kernel=kernel, include_mean=include_mean
        )
        if return_var:
            return cond.loc, cond.variance
        if return_cov:
            return cond.loc, cond.covariance
        return cond.loc

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
            shape = tuple(shape) + (self.num_data,)
        normal_samples = jax.random.normal(key, shape=shape, dtype=self.dtype)
        return self.mean + jnp.einsum(
            "...ij,...j->...i", self.scale_tril, normal_samples
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_log_prob(self, alpha: JAXArray) -> JAXArray:
        loglike = -0.5 * jnp.sum(jnp.square(alpha)) - self.norm
        return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)

    @partial(jax.jit, static_argnums=0)
    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return linalg.solve_triangular(
            self.scale_tril, y - self.loc, lower=True
        )


class ConditionResult(NamedTuple):
    log_probability: JAXArray
    gp: GaussianProcess
