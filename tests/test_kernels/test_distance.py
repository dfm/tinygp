# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from tinygp.kernels import distance


def check(comp, expect, args, order=2, **kwargs):
    np.testing.assert_allclose(expect(*args), comp(*args))
    np.testing.assert_allclose(jax.grad(comp)(*args), jax.grad(expect)(*args))
    check_grads(comp, args, order=order, **kwargs)


def test_l2_distance_grad_at_zero():
    expect = lambda x1, x2: jnp.sqrt(jnp.sum(jnp.square(x1 - x2)))
    comp = distance.L2Distance().distance

    x1 = 0.0
    x2 = 1.5
    check(comp, expect, (x1, x2))

    x1 = jnp.array([0.0, 0.1])
    x2 = jnp.array([1.5, -0.2])
    check(comp, expect, (x1, x2))

    g = jax.grad(comp)(x1, x1)
    np.testing.assert_allclose(expect(x1, x1), comp(x1, x1))
    assert np.all(np.isfinite(g))
