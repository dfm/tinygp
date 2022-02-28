# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["LowerGQSM"]

from typing import Tuple

import jax
import jax.numpy as jnp

from tinygp.helpers import JAXArray, dataclass


@dataclass
class LowerGQSM:
    p: JAXArray
    q: JAXArray
    a: JAXArray
    idx: JAXArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.p.shape[0], self.q.shape[0])

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            q, a, x = data
            fn = a @ f + jnp.outer(q, x)
            return fn, fn

        init = jnp.zeros_like(jnp.outer(self.q[0], x[0]))
        _, f = jax.lax.scan(impl, init, (self.q, self.a, x))
        idx = jnp.clip(self.idx, 0, f.shape[0] - 1)
        mask = jnp.logical_and(self.idx >= 0, self.idx < f.shape[0])
        return jax.vmap(jnp.dot)(jnp.where(mask[:, None], self.p, 0), f[idx])


@dataclass
class UpperGQSM:
    p: JAXArray
    q: JAXArray
    a: JAXArray
    idx: JAXArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.q.shape[0], self.p.shape[0])

    @jax.jit
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            p, a, x = data
            fn = a.T @ f + jnp.outer(p, x)
            return fn, fn

        init = jnp.zeros_like(jnp.outer(self.p[-1], x[-1]))
        _, f = jax.lax.scan(impl, init, (self.p, self.a, x), reverse=True)
        idx = jnp.clip(self.idx, 0, f.shape[0] - 1)
        mask = jnp.logical_and(self.idx >= 0, self.idx < f.shape[0])
        return jax.vmap(jnp.dot)(jnp.where(mask[:, None], self.q, 0), f[idx])
