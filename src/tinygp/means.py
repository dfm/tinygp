# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Mean"]

from typing import Callable, Optional, Union

import jax

from tinygp.helpers import JAXArray
from tinygp.kernels import Kernel


class Mean:
    """A wrapper for the GP mean which supports a constant value or a callable

    In ``tinygp``, a mean function can be any callable which takes as input a
    single coordinate and returns the scalar mean at that location.

    Args:
        value: Either a *scalar* constant, or a callable with the correct
        signature.
    """

    def __init__(self, value: Union[JAXArray, Callable[[JAXArray], JAXArray]]):
        self.value = value

    def __call__(self, X: JAXArray) -> JAXArray:
        if callable(self.value):
            return self.value(X)
        return self.value


class Conditioned:
    """The mean of a process conditioned on observed data

    Args:
        X: The coordinates of the data. alpha: The value :math:`L^-1\,y` where L
        is ``scale_tril`` and y is the
            observed data.
        scale_tril: The lower Cholesky factor of the base process' kernel
            matrix.
        kernel: The predictive kerenl; this will generally be the kernel from
            the kernel used by the original process.
        include_mean: If ``True``, the predicted values will include the mean
            function evaluated at ``X_test``.
        mean_function: The mean function of the base process. Used only if
            ``include_mean`` is ``True``.
    """

    def __init__(
        self,
        X: JAXArray,
        alpha: JAXArray,
        kernel: Kernel,
        *,
        include_mean: bool,
        mean_function: Optional[Mean] = None,
    ):
        self.X = X
        self.alpha = alpha
        self.kernel = kernel
        self.include_mean = include_mean
        self.mean_function = mean_function

    def __call__(self, X: JAXArray) -> JAXArray:
        Ks = jax.vmap(self.kernel.evaluate, in_axes=(None, 0), out_axes=0)(
            X, self.X
        )
        mu = Ks @ self.alpha
        if self.include_mean and self.mean_function is not None:
            mu += self.mean_function(X)
        return mu
