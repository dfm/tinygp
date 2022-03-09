# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.scipy as jsp
import numpy as np
import pytest

from tinygp.kernels import quasisep


@pytest.fixture
def random():
    return np.random.default_rng(84930)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    y = np.sin(x)
    t = np.sort(random.uniform(-3, 3, 12))
    return x, y, t


@pytest.fixture(
    params=[
        quasisep.Matern32(sigma=1.8, scale=1.5),
        quasisep.Matern32(1.5),
        quasisep.Matern52(sigma=1.8, scale=1.5),
        quasisep.Matern52(1.5),
        quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        quasisep.SHO(omega=1.5, quality=0.5, sigma=1.3),
        quasisep.SHO(omega=1.5, quality=3.5, sigma=1.3),
        quasisep.SHO(omega=1.5, quality=0.1, sigma=1.3),
        quasisep.Exp(sigma=1.8, scale=1.5),
        quasisep.Exp(1.5),
        1.5 * quasisep.Matern52(1.5) + 0.3 * quasisep.Exp(1.5),
        quasisep.Matern52(1.5) * quasisep.SHO(omega=1.5, quality=0.1),
        1.5 * quasisep.Matern52(1.5) * quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        quasisep.Cosine(sigma=1.8, scale=1.5),
        1.8 * quasisep.Cosine(1.5),
        quasisep.CARMA.init(
            alpha=np.array([1.4, 2.3, 1.5]), beta=np.array([0.1, 0.5])
        ),
    ]
)
def kernel(request):
    return request.param


def test_quasisep_kernels(data, kernel):
    x, y, t = data
    K = kernel(x, x)

    # Test that to_dense and matmuls work as expected
    np.testing.assert_allclose(kernel.to_symm_qsm(x).to_dense(), K)
    np.testing.assert_allclose(kernel.matmul(x, y), K @ y)
    np.testing.assert_allclose(kernel.matmul(t, x, y), kernel(t, x) @ y)

    # Test that F and are defined consistently
    x1 = x[0]
    x2 = x[1]
    num_A = jsp.linalg.expm(kernel.design_matrix().T * (x2 - x1))
    np.testing.assert_allclose(kernel.transition_matrix(x1, x2), num_A)
