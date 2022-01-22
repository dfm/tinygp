# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np
import pytest

from tinygp import kernels


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    x1 = random.uniform(-3, 3, (50, 5))
    x2 = random.uniform(-5, 5, (50, 5))
    return x1, x2


def test_constant(data):
    x1, x2 = data

    # Check for dimension issues, either when instantiated...
    with pytest.raises(ValueError):
        kernels.Constant(jnp.ones(3))

    # ... or when multiplied
    with pytest.raises(ValueError):
        jnp.ones(3) * kernels.Matern32(1.5)

    # Check that multiplication has the expected behavior
    factor = 2.5
    k1 = kernels.Matern32(2.5)
    np.testing.assert_allclose(factor * k1(x1, x2), (factor * k1)(x1, x2))


def test_custom(data):
    x1, x2 = data

    # Check that known kernels work as expected
    scale = 1.5
    k1 = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.sum(jnp.square((X1 - X2) / scale)))
    )
    k2 = kernels.ExpSquared(scale)
    np.testing.assert_allclose(k1(x1, x2), k2(x1, x2))

    scale = 1.5 * jnp.ones(x1.shape[1])
    k1 = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.sum(jnp.square((X1 - X2) / scale)))
    )
    k2 = kernels.ExpSquared(scale)
    np.testing.assert_allclose(k1(x1, x2), k2(x1, x2))

    # Check that an invalid kernel raises as expected
    kernel = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.square((X1 - X2) / scale))
    )
    with pytest.raises(ValueError):
        kernel(x1, x2)