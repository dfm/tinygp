# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    # "factor",
    # "solve_lower",
    # "solve_upper",
    "matmul_lower",
    # "matmul_upper",
]

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import lax

from tinygp.types import JAXArray

Propagator = Callable[[JAXArray, JAXArray, JAXArray], JAXArray]


def matmul_lower(
    propagate: Propagator,
    X1: JAXArray,
    X2: JAXArray,
    U: JAXArray,
    V: JAXArray,
    Y: JAXArray,
) -> JAXArray:
    def impl(carry, data):
        (Fp, Vp, Yp) = carry
        Vn, Xp, Xn, Yn = data
        Fn = propagate(Xp, Xn, Fp + jnp.outer(Yp, Vp))
        return (Fn, Vn, Yn), Fn

    J = V.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((M, J), dtype=V.dtype)
    Vi = jnp.zeros_like(V[0])
    Yi = jnp.zeros_like(Y[0])
    F = lax.scan(impl, (Fi, Vi, Yi), (V, X1, X2, Y))[1]

    return jnp.einsum("nj,nkj->nk", U, F)
