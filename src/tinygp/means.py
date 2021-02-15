# -*- coding: utf-8 -*-

__all__ = ["zero_mean", "constant_mean"]

from typing import Callable

import jax.numpy as jnp

Mean = Callable[[jnp.ndarray], jnp.ndarray]


def zero_mean(X: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(X.shape[0])


def constant_mean(value: jnp.ndarray) -> Mean:
    _value = value

    def mean(X: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(X.shape[0], _value)

    return mean
