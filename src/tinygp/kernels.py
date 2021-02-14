# -*- coding: utf-8 -*-

__all__ = [
    "metric",
    "diagonal_metric",
    "dense_metric",
    "cholesky_metric",
    "constant_kernel",
    "exp_squared_kernel",
]

from functools import partial

import jax.numpy as jnp
from jax.scipy import linalg

from .functional import compose


def metric(r):
    return jnp.sum(jnp.square(r))


def diagonal_metric(ell):
    return partial(jnp.multiply, 1.0 / ell)


def dense_metric(cov, *, lower=True):
    chol = linalg.cholesky(cov, lower=lower)
    return cholesky_metric(chol, lower=lower)


def cholesky_metric(chol, *, lower=True):
    solve = partial(linalg.solve_triangular, chol, lower=lower)
    return compose(metric, solve)


def constant_kernel(value):
    _value = value
    return lambda x1, x2: _value


def exp_squared_kernel(metric):
    return compose(lambda x: jnp.exp(-0.5 * x), metric)
