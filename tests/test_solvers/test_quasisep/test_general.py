# -*- coding: utf-8 -*-
# mypy: ignore-errors

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinygp.solvers.quasisep.general import LowerGQSM, UpperGQSM


def test_lower_matmul():
    random = np.random.default_rng(1234)

    sigma = 1.5
    scale = 3.4
    f = np.sqrt(3) / scale

    @jax.vmap
    def get_a(dt):
        return jnp.exp(-f * dt) * jnp.array(
            [[1 + f * dt, dt], [-jnp.square(f) * dt, 1 - f * dt]]
        )

    x1 = np.sort(random.uniform(0, 10, 100))
    x2 = np.sort(random.uniform(2, 8, 75))

    for (x1, x2) in [(x1, x2), (x1, x1), (x2, x1)]:
        idx = np.searchsorted(x2, x1, side="right") - 1
        y = np.sin(x2)[:, None]

        r = np.abs(x1[:, None] - x2[None, :]) / scale
        arg = np.sqrt(3) * r
        K = sigma ** 2 * (1 + arg) * np.exp(-arg)
        K[x1[:, None] < x2[None, :]] = 0.0

        a = get_a(jnp.append(0, jnp.diff(x2)))
        q = jnp.stack((jnp.ones_like(x2), jnp.zeros_like(x2)), axis=-1)
        p = sigma ** 2 * get_a(x1 - x2[idx])[:, 0, :]
        m = LowerGQSM(p=p, q=q, a=a, idx=idx)
        np.testing.assert_allclose(m.matmul(y), K @ y)


def test_upper_matmul():
    random = np.random.default_rng(1234)

    sigma = 1.5
    scale = 3.4
    f = np.sqrt(3) / scale

    @jax.vmap
    def get_a(dt):
        return jnp.exp(-f * dt) * jnp.array(
            [[1 + f * dt, dt], [-jnp.square(f) * dt, 1 - f * dt]]
        )

    x1 = np.sort(random.uniform(2, 8, 75))
    x2 = np.sort(random.uniform(0, 10, 100))
    # x2 = x1

    # for (x1, x2) in [(x1, x2), (x1, x1), (x2, x1)]:
    idx = np.searchsorted(x2, x1, side="right") - 1
    y = np.sin(x2)[:, None]

    r = np.abs(x1[:, None] - x2[None, :]) / scale
    arg = np.sqrt(3) * r
    K = sigma ** 2 * (1 + arg) * np.exp(-arg)
    K[x1[:, None] >= x2[None, :]] = 0.0

    a = get_a(jnp.append(0, jnp.diff(x2)))
    p = a[:, 0, :]
    # p = jnp.stack((jnp.ones_like(x2), jnp.zeros_like(x2)), axis=-1)
    q = sigma ** 2 * get_a(x2[idx] - x1)[:, :, 0]
    m = UpperGQSM(p=p, q=q, a=a, idx=idx)
    # print(m.matmul(y)[:, 0])
    # print((K @ y)[:, 0])
    # assert 0
    np.testing.assert_allclose(m.matmul(y), K @ y)
