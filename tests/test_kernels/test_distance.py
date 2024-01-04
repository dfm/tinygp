# mypy: ignore-errors

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from tinygp.kernels import distance
from tinygp.test_utils import assert_allclose


def check(comp, expect, args, order=2, **kwargs):
    assert_allclose(expect(*args), comp(*args))
    assert_allclose(jax.grad(comp)(*args), jax.grad(expect)(*args))
    check_grads(comp, args, order=order, **kwargs)


def test_l2_distance_grad_at_zero():
    def expect(x1, x2):
        return jnp.sqrt(jnp.sum(jnp.square(x1 - x2)))

    comp = distance.L2Distance().distance

    x1 = 0.0
    x2 = 1.5
    check(comp, expect, (x1, x2))

    x1 = jnp.array([0.0, 0.1])
    x2 = jnp.array([1.5, -0.2])
    check(comp, expect, (x1, x2))

    g = jax.grad(comp)(x1, x1)
    assert_allclose(expect(x1, x1), comp(x1, x1))
    assert jnp.all(jnp.isfinite(g))
