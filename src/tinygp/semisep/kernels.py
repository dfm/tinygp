# -*- coding: utf-8 -*-

from __future__ import annotations
from this import d

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

    def propagate(self, dX: JAXArray, state: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def matmul(self, X: JAXArray, Y: JAXArray) -> JAXArray:
        U = jax.vmap(self.left)(X)
        V = jax.vmap(self.right)(X)
        return (
            jnp.einsum("nj,nj,n...->n...", U, V, Y)
            + linalg.matmul_lower(self.propagate, X, U, V, Y)
            + linalg.matmul_upper(self.propagate, X, U, V, Y)
        )

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.where(
            X1 >= X2,
            self.left(X1).T @ self.propagate(X1 - X2, self.right(X2)),
            self.left(X2).T @ self.propagate(X2 - X1, self.right(X1)),
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

    def propagate(self, dt: JAXArray, state: JAXArray) -> JAXArray:
        if jnp.ndim(dt) != 0:
            raise ValueError("Only 1D inputs are supported")
        factor = jnp.exp(-jnp.append(self.c, self.c) * dt)
        return factor * state
