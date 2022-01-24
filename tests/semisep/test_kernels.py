# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np
import pytest

from tinygp.semisep import kernels


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    x1 = np.sort(random.uniform(-3, 3, 50))
    x2 = np.sort(random.uniform(-5, 5, 45))
    y = random.normal(size=(len(x1), 5))
    return x1, x2, y


@pytest.mark.parametrize(
    "args",
    [
        (1.5, 0.8, 2.4, 1.3),
        (
            jnp.array([1.5, 1.7]),
            jnp.array([0.8, 0.7]),
            jnp.array([2.4, 2.6]),
            jnp.array([1.3, 0.9]),
        ),
    ],
)
def test_celerite_nd(data, args):
    x1, x2, y = data
    a, b, c, d = args
    kernel = kernels.Celerite(a, b, c, d)

    # Evaluate the kernel function directly
    tau = jnp.abs(x1[:, None] - x2[None, :])[:, :, None]
    expected = jnp.sum(
        jnp.exp(-c * tau) * (a * jnp.cos(d * tau) + b * jnp.sin(d * tau)),
        axis=-1,
    )

    # Check that invalid (multivariate) data raises
    with pytest.raises(ValueError):
        kernel(jnp.repeat(x1[:, None], 2, -1), jnp.repeat(x2[:, None], 2, -1))

    # Check that evaluate is defined properly
    np.testing.assert_allclose(kernel(x1, x2), expected, atol=1e-6)

    # Check the matmul operation
    actual = kernel.matmul(x1, y)
    expected = np.tril(kernel(x1), -1) @ y
    print(actual)
    print(expected)
    np.testing.assert_allclose(actual, expected)
