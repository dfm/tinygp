# -*- coding: utf-8 -*-

__all__ = [
    "metric",
    "diagonal_metric",
    "dense_metric",
    "cholesky_metric",
    "Constant",
    "ExpSquared",
]

from typing import Callable, Union

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from .functional import compose


Metric = Callable[[jnp.ndarray], jnp.ndarray]


def metric(r: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(r))


def diagonal_metric(ell: jnp.ndarray) -> Metric:
    return partial(jnp.multiply, 1.0 / ell)


def dense_metric(cov: jnp.ndarray, *, lower: bool = True) -> Metric:
    chol = linalg.cholesky(cov, lower=lower)
    return cholesky_metric(chol, lower=lower)


def cholesky_metric(chol: jnp.ndarray, *, lower: bool = True) -> Metric:
    solve = partial(linalg.solve_triangular, chol, lower=lower)
    return compose(metric, solve)


class Kernel:
    def evaluate_diag(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self)(X, X)

    def evaluate(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda _X1: jax.vmap(lambda _X2: self(_X1, _X2))(X2))(
            X1
        )

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


class Constant(Kernel):
    def __init__(self, value: jnp.ndarray):
        self.value = jnp.asarray(value)

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return self.value


class MetricKernel(Kernel):
    def __init__(self, metric: Union[Metric, jnp.ndarray]):
        if callable(metric):
            self.metric = metric
        else:
            self.metric = diagonal_metric(metric)

    def evaluate_radial(self, r2: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return self.evaluate_radial(self.metric(X1 - X2))


class ExpSquared(MetricKernel):
    def evaluate_radial(self, r2: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * r2)


class Exp(MetricKernel):
    def evaluate_radial(self, r2: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-jnp.sqrt(r2))
