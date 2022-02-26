# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy import linalg

from tinygp import kernels as base_kernels
from tinygp.solvers.quasisep import kernels


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    return x


def test_matern32(data):
    kernel = kernels.Matern32(1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data), atol=1e-6
    )

    kernel = 2.3 * kernels.Matern32(1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data), atol=1e-6
    )
