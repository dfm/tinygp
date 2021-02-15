# -*- coding: utf-8 -*-

__all__ = [
    "compose",
    "unit_metric",
    "diagonal_metric",
    "dense_metric",
    "cholesky_metric",
]

from typing import Callable
from functools import reduce, partial

import jax.numpy as jnp
from jax.scipy import linalg


Metric = Callable[[jnp.ndarray], jnp.ndarray]


def compose(*functions):
    return reduce(lambda f, g: lambda *args: f(g(*args)), functions)


def unit_metric(r: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(r))


def diagonal_metric(ell: jnp.ndarray) -> Metric:
    return compose(unit_metric, partial(jnp.multiply, 1.0 / ell))


def dense_metric(cov: jnp.ndarray, *, lower: bool = True) -> Metric:
    chol = linalg.cholesky(cov, lower=lower)
    return cholesky_metric(chol, lower=lower)


def cholesky_metric(chol: jnp.ndarray, *, lower: bool = True) -> Metric:
    solve = partial(linalg.solve_triangular, chol, lower=lower)
    return compose(unit_metric, solve)
