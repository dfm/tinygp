# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["GaussianProcess"]

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp

from tinygp import kernels, means
from tinygp.helpers import JAXArray
from tinygp.kernels.quasisep import Quasisep
from tinygp.solvers import DirectSolver, QuasisepSolver
from tinygp.solvers.quasisep.core import SymmQSM

if TYPE_CHECKING:
    from tinygp.numpyro_support import TinyDistribution


class GaussianProcess:
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function X (JAXArray): The input
            coordinates. This can be any PyTree that is compatible with
            ``kernel`` where the zeroth dimension is ``N_data``, the size of the
            data set.
        diag (JAXArray, optional): The value to add to the diagonal of the
            covariance matrix, often used to capture measurement uncertainty.
            This should be a scalar or have the shape ``(N_data,)``. If not
            provided, this will default to the square root of machine epsilon
            for the data type being used. This can sometimes be sufficient to
            avoid numerical issues, but if you're getting NaNs, try increasing
            this value.
        mean (Callable, optional): A callable or constant mean function that
            will be evaluated with the ``X`` as input: ``mean(X)``
        mean_value (JAXArray, optional): The mean precomputed at the location
            of the data.
        covariance_value (JAXArray, optional): The covariance matrix precomputed
            at the location of the data.
        solver: The solver type to be used to execute the required linear
            algebra.
    """

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        *,
        diag: Optional[JAXArray] = None,
        mean: Optional[Union[Callable[[JAXArray], JAXArray], JAXArray]] = None,
        mean_value: Optional[JAXArray] = None,
        covariance_value: Optional[Any] = None,
        solver: Optional[Any] = None,
    ):
        self.kernel = kernel
        self.X = X

        if callable(mean):
            self.mean_function = mean
        else:
            if mean is None:
                self.mean_function = means.Mean(jnp.zeros(()))
            else:
                self.mean_function = means.Mean(mean)
        if mean_value is None:
            mean_value = jax.vmap(self.mean_function)(self.X)
        self.num_data = mean_value.shape[0]
        self.dtype = mean_value.dtype
        self.loc = self.mean = mean_value
        if self.mean.ndim != 1:
            raise ValueError(
                "Invalid mean shape: "
                f"expected ndim = 1, got ndim={self.mean.ndim}"
            )

        self.diag = _default_diag(self.mean) if diag is None else diag
        if solver is None:
            if isinstance(covariance_value, SymmQSM) or isinstance(
                kernel, Quasisep
            ):
                solver = QuasisepSolver
            else:
                solver = DirectSolver
        self.solver = solver.init(
            kernel, self.X, self.diag, covariance=covariance_value
        )

    @property
    def variance(self) -> JAXArray:
        return self.solver.variance()

    @property
    def covariance(self) -> JAXArray:
        return self.solver.covariance()

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

        alpha, log_prob, mean_value = self._condition(
            y, X_test, include_mean, kernel
        )
        if kernel is None:
            kernel = self.kernel

        diag = _default_diag(self.mean) if diag is None else diag
        covariance_value = self.solver.condition(kernel, X_test, diag)
        if X_test is None:
            X_test = self.X

        # The conditional GP will also be a GP with the mean an covariance
        # specified by a :class:`tinygp.means.Conditioned` and
        # :class:`tinygp.kernels.Conditioned` respectively.
        gp = GaussianProcess(
            kernels.Conditioned(self.X, self.solver, kernel),
            X_test,
            diag=diag,
            mean=means.Conditioned(
                self.X,
                alpha,
                kernel,
                include_mean=include_mean,
                mean_function=self.mean_function,
            ),
            mean_value=mean_value,
            covariance_value=covariance_value,
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

    def numpyro_dist(self, **kwargs: Any) -> "TinyDistribution":
        """Get the numpyro MultivariateNormal distribution for this process"""
        from tinygp.numpyro_support import TinyDistribution

        return TinyDistribution(self, **kwargs)  # type: ignore

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
        return self.mean + jnp.moveaxis(
            self.solver.dot_triangular(normal_samples), 0, -1
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_log_prob(self, alpha: JAXArray) -> JAXArray:
        loglike = (
            -0.5 * jnp.sum(jnp.square(alpha)) - self.solver.normalization()
        )
        return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)

    @partial(jax.jit, static_argnums=0)
    def _get_alpha(self, y: JAXArray) -> JAXArray:
        return self.solver.solve_triangular(y - self.loc)

    @partial(jax.jit, static_argnums=(0, 3))
    def _condition(
        self,
        y: JAXArray,
        X_test: Optional[JAXArray],
        include_mean: bool,
        kernel: Optional[kernels.Kernel] = None,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        alpha = self._get_alpha(y)
        log_prob = self._compute_log_prob(alpha)

        # Below, we actually want alpha = K^-1 y instead of alpha = L^-1 y
        alpha = self.solver.solve_triangular(alpha, transpose=True)

        if X_test is None:
            X_test = self.X

            # In this common case (where we're predicting the GP at the data
            # points, using the original kernel), the mean is especially fast to
            # compute; so let's use that calculation here.
            if kernel is None:
                delta = self.diag * alpha
                mean_value = y - delta
                if not include_mean:
                    mean_value -= self.loc

            else:
                mean_value = kernel.matmul(self.X, y=alpha)
                if include_mean:
                    mean_value += self.loc

        else:
            if kernel is None:
                kernel = self.kernel

            mean_value = kernel.matmul(X_test, self.X, alpha)
            if include_mean:
                mean_value += jax.vmap(self.mean_function)(X_test)

        return alpha, log_prob, mean_value


class ConditionResult(NamedTuple):
    """The result of conditioning a :class:`GaussianProcess` on data

    This has two entries, ``log_probability`` and ``gp``, that are described
    below.
    """

    log_probability: JAXArray
    """The log probability of the conditioned model

    In other words, this is the marginal likelihood for the kernel parameters,
    given the observed data, or the multivariate normal log probability
    evaluated at the given data.
    """

    gp: GaussianProcess
    """A :class:`GaussianProcess` describing the conditional distribution

    This will have a mean and covariance conditioned on the observed data, but
    it is otherwise a fully functional GP that can sample from or condition
    further (although that's probably not going to be very efficient).
    """


def _default_diag(reference: JAXArray) -> JAXArray:
    """Default to adding some amount of jitter to the diagonal, just in case,
    we use sqrt(eps) for the dtype of the mean function because that seems to
    give sensible results in general.
    """
    return jnp.sqrt(jnp.finfo(reference).eps)  # type: ignore
