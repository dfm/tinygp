# -*- coding: utf-8 -*-
# mypy: ignore-errors

import numpy as np
import pytest

from tinygp.solvers.quasisep import kernels


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    return x


def test_matern32(data):
    kernel = kernels.Matern32(sigma=1.8, scale=1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )

    kernel = 2.3 * kernels.Matern32(1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )


def test_matern52(data):
    kernel = kernels.Matern52(sigma=1.8, scale=1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )

    kernel = 2.3 * kernels.Matern52(1.5)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )


def test_celerite(data):
    kernel = kernels.Celerite(1.1, 0.8, 0.9, 0.1)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )

    kernel = 2.3 * kernels.Celerite(1.1, 0.8, 0.9, 0.1)
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )
