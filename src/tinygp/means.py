# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["zero_mean", "constant_mean"]

from typing import Callable

import jax.numpy as jnp

from .types import JAXArray

Mean = Callable[[JAXArray], JAXArray]


def zero_mean(X: JAXArray) -> JAXArray:
    return jnp.zeros(())


def constant_mean(value: JAXArray) -> Mean:
    _value = value

    def mean(X: JAXArray) -> JAXArray:
        return _value

    return mean
