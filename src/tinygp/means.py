# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Mean"]

from typing import Callable, Union

import jax.numpy as jnp

from .types import JAXArray


class Mean:
    def __init__(self, value: Union[JAXArray, Callable[[JAXArray], JAXArray]]):
        self.value = value

    def __call__(self, X: JAXArray) -> JAXArray:
        if callable(self.value):
            return self.value(X)
        return self.value


class Conditioned(Mean):
    pass
