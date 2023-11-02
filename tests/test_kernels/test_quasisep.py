# mypy: ignore-errors

import jax
import jax.scipy as jsp
import numpy as np
import pytest

from tinygp import GaussianProcess
from tinygp.kernels import quasisep

jax.config.update("jax_enable_x64", True)


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
        quasisep.CARMA.init(alpha=[1.4, 2.3, 1.5], beta=[0.1, 0.5]),
        quasisep.CARMA.init(alpha=[1.4, 2.3, 1.5], beta=[0.1, 0.5]),
        quasisep.CARMA.init(alpha=[1, 1.2], beta=[1.0, 3.0]),
        quasisep.CARMA.init(alpha=[0.1, 1.1], beta=[1.0, 3.0]),
        quasisep.CARMA.init(alpha=[1.0 / 100], beta=[0.3]),
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


def test_celerite(data):
    a, b, c, d = 1.1, 0.8, 0.9, 0.1
    kernel = quasisep.Celerite(a, b, c, d)

    x, _, t = data

    calc = kernel(x, x)
    tau = np.abs(x[:, None] - x[None, :])
    expect = np.exp(-c * tau) * (a * np.cos(d * tau) + b * np.sin(d * tau))
    np.testing.assert_allclose(calc, expect)

    calc = kernel(x, t)
    tau = np.abs(x[:, None] - t[None, :])
    expect = np.exp(-c * tau) * (a * np.cos(d * tau) + b * np.sin(d * tau))
    np.testing.assert_allclose(calc, expect)


def test_carma(data):
    x, y, t = data
    # CARMA kernels
    carma2_kernels = [
        quasisep.CARMA.init(alpha=[0.01], beta=[0.1]),
        quasisep.CARMA.init(alpha=[1.0, 1.2], beta=[1.0, 3.0]),
        quasisep.CARMA.init(alpha=[0.1, 1.1], beta=[1.0, 3.0]),
    ]
    # Equivalent Celerite+Exp kernels for validation
    validate_kernels = [
        quasisep.Exp(scale=100.0, sigma=np.sqrt(0.5)),
        quasisep.Celerite(25.0 / 6, 2.5, 0.6, -0.8),
        quasisep.Exp(1.0, np.sqrt(4.04040404))
        + quasisep.Exp(10.0, np.sqrt(4.5959596)),
    ]

    # Compare log_probability & normalization
    for i in range(len(carma2_kernels)):
        gp1 = GaussianProcess(carma2_kernels[i], x, diag=0.1)
        gp2 = GaussianProcess(validate_kernels[i], x, diag=0.1)

        np.testing.assert_allclose(
            gp1.log_probability(y), gp2.log_probability(y)
        )
        np.testing.assert_allclose(
            gp1.solver.normalization(), gp2.solver.normalization()
        )


def test_carma_jit(data):
    x, y, t = data

    def build_gp(params):
        carma_kernel = quasisep.CARMA.init(
            alpha=params["alpha"], beta=params["beta"]
        )
        return GaussianProcess(carma_kernel, x, diag=0.01, mean=0.0)

    @jax.jit
    def loss(params):
        gp = build_gp(params)
        return -gp.log_probability(y)

    params = {"alpha": [1.0, 1.2], "beta": [1.0, 3.0]}
    loss(params)


def test_carma_fpoly():
    alpha = np.array([1.4, 2.3, 1.5])
    beta = np.array([0.1, 0.5])
    alpha_fpoly, alpha_mult = quasisep.CARMA.poly2fpoly(np.append(alpha, 1.0))
    beta_fpoly, beta_mult = quasisep.CARMA.poly2fpoly(beta)

    carma31 = quasisep.CARMA.init(alpha=alpha, beta=beta)
    carma31_fpoly = quasisep.CARMA.from_fpoly(
        alpha_fpoly=alpha_fpoly, beta_fpoly=beta_fpoly, beta_mult=beta_mult
    )

    # if two constructor give the same model
    assert np.allclose(carma31.arroots, carma31_fpoly.arroots)
    assert np.allclose(carma31.acf, carma31_fpoly.acf)
    assert np.allclose(carma31.obsmodel, carma31_fpoly.obsmodel)
