# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["SemisepKernel", "Celerite"]

import jax
import jax.numpy as jnp

from tinygp.kernels import Kernel
from tinygp.types import JAXArray
from tinygp.semisep import linalg


class SemisepKernel(Kernel):
    def left(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def right(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def propagate(
        self, X1: JAXArray, X2: JAXArray, state: JAXArray
    ) -> JAXArray:
        raise NotImplementedError()

    def matmul(self, X: JAXArray, Y: JAXArray) -> JAXArray:
        U = jax.vmap(self.left)(X)
        V = jax.vmap(self.right)(X)
        Xn = jax.tree_map(lambda arg: jnp.append(arg[1:], arg[-1]), X)
        return linalg.matmul_lower(self.propagate, X, Xn, U, V, Y)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.where(
            X1 >= X2,
            self.left(X1).T @ self.propagate(X1, X2, self.right(X2)),
            self.left(X2).T @ self.propagate(X2, X1, self.right(X1)),
        )


class Celerite(SemisepKernel):
    def __init__(self, a: JAXArray, b: JAXArray, c: JAXArray, d: JAXArray):
        self.a, self.b, self.c, self.d = jnp.broadcast_arrays(a, b, c, d)
        if jnp.ndim(self.a) > 1:
            raise ValueError("Only scalar or 1-D coefficients are supported")

    def left(self, t: JAXArray) -> JAXArray:
        if jnp.ndim(t) != 0:
            raise ValueError("Only 1D inputs are supported")
        arg = self.d * t
        cos = jnp.cos(arg)
        sin = jnp.sin(arg)
        return jnp.append(
            self.a * cos + self.b * sin, self.a * sin - self.b * cos
        )

    def right(self, t: JAXArray) -> JAXArray:
        if jnp.ndim(t) != 0:
            raise ValueError("Only 1D inputs are supported")
        arg = self.d * t
        return jnp.append(jnp.cos(arg), jnp.sin(arg))

    def propagate(
        self, t1: JAXArray, t2: JAXArray, state: JAXArray
    ) -> JAXArray:
        if jnp.ndim(t1) != 0 or jnp.ndim(t2) != 0:
            raise ValueError("Only 1D inputs are supported")
        factor = jnp.exp(jnp.append(self.c, self.c) * (t2 - t1))
        # print(factor)
        return factor * state
