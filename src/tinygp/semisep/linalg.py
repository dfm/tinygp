# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    # "factor",
    # "solve_lower",
    # "solve_upper",
    "matmul_lower",
    "matmul_upper",
]

from typing import Callable

import jax.numpy as jnp
from jax import lax, tree_map

from tinygp.types import JAXArray

Propagator = Callable[[JAXArray, JAXArray, JAXArray], JAXArray]


def matmul_lower(
    propagate: Propagator,
    X: JAXArray,
    U: JAXArray,
    V: JAXArray,
    Y: JAXArray,
) -> JAXArray:
    def impl(carry, data):  # type: ignore
        Fp, Xp, Vp, Yp = carry
        Xn, Vn, Yn = data
        Fn = propagate(Xp, Xn, Fp + jnp.outer(Yp, Vp))
        return (Fn, Xn, Vn, Yn), Fn

    Vi = jnp.zeros_like(V[0])
    Yi = jnp.zeros_like(Y[0])
    Fi = jnp.zeros_like(jnp.outer(Yi, Vi))
    Xi = tree_map(lambda arg: arg[0], X)
    F = lax.scan(impl, (Fi, Xi, Vi, Yi), (X, V, Y))[1]
    return jnp.einsum("nj,nkj->nk", U, F)


def matmul_upper(
    propagate: Propagator,
    X: JAXArray,
    U: JAXArray,
    V: JAXArray,
    Y: JAXArray,
) -> JAXArray:
    def impl(carry, data):  # type: ignore
        Fp, Xp, Up, Yp = carry
        Xn, Un, Yn = data
        Fn = propagate(Xn, Xp, Fp + jnp.outer(Yp, Up))
        return (Fn, Xn, Un, Yn), Fn

    Ui = jnp.zeros_like(U[0])
    Yi = jnp.zeros_like(Y[0])
    Fi = jnp.zeros_like(jnp.outer(Yi, Ui))
    Xi = tree_map(lambda arg: arg[-1], X)
    F = lax.scan(impl, (Fi, Xi, Ui, Yi), (X, U, Y), reverse=True)[1]
    return jnp.einsum("nj,nkj->nk", V, F)
