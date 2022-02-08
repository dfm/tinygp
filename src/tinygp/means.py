# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Mean"]

from typing import Callable, Optional, Union

import jax.numpy as jnp
from jax.scipy import linalg

from tinygp.kernels import Kernel
from tinygp.types import JAXArray


class Mean:
    def __init__(self, value: Union[JAXArray, Callable[[JAXArray], JAXArray]]):
        self.value = value

    def __call__(self, X: JAXArray) -> JAXArray:
        if callable(self.value):
            return self.value(X)
        return self.value


class Conditioned:
    def __init__(
        self,
        X: JAXArray,
        alpha: JAXArray,
        scale_tril: JAXArray,
        kernel: Kernel,
        *,
        include_mean: bool,
        mean_function: Optional[Mean] = None,
    ):
        self.X = X
        self.alpha = alpha
        self.scale_tril = scale_tril
        self.kernel = kernel
        self.include_mean = include_mean
        self.mean_function = mean_function

    def __call__(self, X: JAXArray) -> JAXArray:
        mu = self.alpha @ linalg.solve_triangular(
            self.scale_tril,
            self.kernel(self.X, X),
            lower=True,
        )
        if self.include_mean and self.mean_function is not None:
            mu += self.mean_function(X)
        return mu
