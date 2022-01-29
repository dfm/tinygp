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

Propagator = Callable[[JAXArray, JAXArray], JAXArray]


def matmul_lower(
    propagate: Propagator,
    X: JAXArray,
    U: JAXArray,
    V: JAXArray,
    Y: JAXArray,
) -> JAXArray:
    def impl(carry, data):  # type: ignore
        Fp, Vp, Yp = carry
        dX, Vn, Yn = data
        Fn = propagate(dX, Fp + jnp.outer(Yp, Vp))
        return (Fn, Vn, Yn), Fn

    J = V.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((M, J), dtype=V.dtype)
    Vi = jnp.zeros_like(V[0])
    Yi = jnp.zeros_like(Y[0])
    dX = tree_map(lambda arg: jnp.append(0, arg[1:] - arg[:-1]), X)
    F = lax.scan(impl, (Fi, Vi, Yi), (dX, V, Y))[1]

    return jnp.einsum("nj,nkj->nk", U, F)


def matmul_upper(
    propagate: Propagator,
    X: JAXArray,
    U: JAXArray,
    V: JAXArray,
    Y: JAXArray,
) -> JAXArray:
    def impl(carry, data):  # type: ignore
        Fp, Up, Yp = carry
        dX, Un, Yn = data
        Fn = propagate(dX, Fp + jnp.outer(Yp, Up))
        return (Fn, Un, Yn), Fn

    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((M, J), dtype=V.dtype)
    Ui = jnp.zeros_like(U[0])
    Yi = jnp.zeros_like(Y[0])
    dX = tree_map(lambda arg: jnp.append(arg[1:] - arg[:-1], 0), X)
    F = lax.scan(impl, (Fi, Ui, Yi), (dX, U, Y), reverse=True)[1]

    return jnp.einsum("nj,nkj->nk", V, F)
