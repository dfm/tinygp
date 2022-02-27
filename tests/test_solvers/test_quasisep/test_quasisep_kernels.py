# -*- coding: utf-8 -*-
# mypy: ignore-errors

import numpy as np
import pytest

from tinygp.solvers.quasisep import kernels


@pytest.fixture
def random():
    return np.random.default_rng(84930)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    return x


@pytest.fixture(
    params=[
        kernels.Matern32(sigma=1.8, scale=1.5),
        kernels.Matern32(1.5),
        kernels.Matern52(sigma=1.8, scale=1.5),
        kernels.Matern52(1.5),
        kernels.Celerite(1.1, 0.8, 0.9, 0.1),
        kernels.SHO(omega=1.5, quality=0.5, sigma=1.3),
        kernels.SHO(omega=1.5, quality=3.5, sigma=1.3),
        kernels.SHO(omega=1.5, quality=0.1, sigma=1.3),
        kernels.Exp(sigma=1.8, scale=1.5),
        kernels.Exp(1.5),
        1.5 * kernels.Matern52(1.5) + 0.3 * kernels.Exp(1.5),
        1.5 * kernels.Matern52(1.5) * kernels.Celerite(1.1, 0.8, 0.9, 0.1),
    ]
)
def kernel(request):
    return request.param


def test_to_qsm(data, kernel):
    np.testing.assert_allclose(
        kernel.to_qsm(data).to_dense(), kernel(data, data)
    )
