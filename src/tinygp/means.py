# -*- coding: utf-8 -*-

__all__ = ["zero_mean", "constant_mean"]

from typing import Callable

import jax.numpy as jnp

from .types import JAXArray

Mean = Callable[[JAXArray], JAXArray]


def zero_mean(X: JAXArray) -> JAXArray:
    return jnp.zeros(X.shape[0])


def constant_mean(value: JAXArray) -> Mean:
    _value = value

    def mean(X: JAXArray) -> JAXArray:
        return jnp.full(X.shape[0], _value)

    return mean
