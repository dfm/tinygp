# -*- coding: utf-8 -*-

from __future__ import annotations

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


def unit_metric(r: JAXArray) -> JAXArray:
    """Compute the squared norm of a vector

    This corresponds to the trivial metric with a unit length scale.

    Args:
        r (JAXArray): A coordinate vector

    Returns:
        JAXArray: The transformed vector
    """
    return r


def diagonal_metric(ell: JAXArray) -> Metric:
    """A diagonal metric with a length scale per dimension

    Args:
        ell (JAXArray): The length scale for each dimension. If this is a
            scalar, the metric will be isotropic.
    """
    return partial(jnp.multiply, 1.0 / ell)


def dense_metric(cov: JAXArray) -> Metric:
    """A full-rank general metric

    The units of the covariance parameter are length^2, unlike the other
    metrics. Therefore,

    .. code-block:: python

        dense_metric(jnp.diag(jnp.square(ell)))

    and

    .. code-block:: python

        diagonal_metric(ell)

    are equivalent.

    Args:
        cov (JAXArray): The covariance matrix metric. This must be positive
            definite.
    """
    chol = linalg.cholesky(cov, lower=True)
    return cholesky_metric(chol, lower=True)


def cholesky_metric(chol: JAXArray, *, lower: bool = True) -> Metric:
    """A general metric parameterized by its Cholesky factor

    The units of the Cholesky factor are length, unlike the dense metric.
    Therefore,

    .. code-block:: python

        cholesky_metric(jnp.diag(ell))

    and

    .. code-block:: python

        diagonal_metric(ell)

    are equivalent.

    Args:
        chol (JAXArray): The covariance matrix metric. This must be positive
            definite.
        lower (bool, optional): Is ``chol`` lower triangular? This must be
            ``True``, but is included as a sanity check.
    """
    assert lower
    return partial(linalg.solve_triangular, chol, lower=lower)


def compose(*functions: Metric) -> Metric:
    """A helper function for composing metrics"""
    return reduce(lambda f, g: lambda *args: f(g(*args)), functions)
