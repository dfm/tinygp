# -*- coding: utf-8 -*-

__all__ = [
    "compose",
    "unit_metric",
    "diagonal_metric",
    "dense_metric",
    "cholesky_metric",
]

from functools import partial, reduce
from typing import Callable

import jax.numpy as jnp
from jax.scipy import linalg

from .types import JAXArray

Metric = Callable[[JAXArray], JAXArray]


def compose(*functions):
    return reduce(lambda f, g: lambda *args: f(g(*args)), functions)


def unit_metric(r: JAXArray) -> JAXArray:
    return jnp.sum(jnp.square(r))


def diagonal_metric(ell: JAXArray) -> Metric:
    return compose(unit_metric, partial(jnp.multiply, 1.0 / ell))


def dense_metric(cov: JAXArray, *, lower: bool = True) -> Metric:
    chol = linalg.cholesky(cov, lower=lower)
    return cholesky_metric(chol, lower=lower)


def cholesky_metric(chol: JAXArray, *, lower: bool = True) -> Metric:
    solve = partial(linalg.solve_triangular, chol, lower=lower)
    return compose(unit_metric, solve)
