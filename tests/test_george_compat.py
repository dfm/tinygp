# -*- coding: utf-8 -*-

import numpy as np
import pytest
from jax.config import config

from tinygp import GaussianProcess, kernels, metrics

george = pytest.importorskip("george")

config.update("jax_enable_x64", True)


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture(
    scope="module",
    params=["Constant", "DotProduct", "Polynomial", "Linear"],
)
def kernel(request):
    return {
        "Constant": (
            kernels.Constant(value=1.5),
            george.kernels.ConstantKernel(
                log_constant=np.log(1.5 / 5), ndim=5
            ),
        ),
        "DotProduct": (
            kernels.DotProduct(),
            george.kernels.DotProductKernel(ndim=3),
        ),
        "Polynomial": (
            kernels.Polynomial(order=2.5, sigma=1.3),
            george.kernels.PolynomialKernel(
                order=2.5, log_sigma2=2 * np.log(1.3), ndim=1
            ),
        ),
        "Linear": (
            kernels.Linear(order=2.5, sigma=1.3),
            george.kernels.LinearKernel(
                order=2.5, log_gamma2=2.5 * 2 * np.log(1.3), ndim=1
            ),
        ),
    }[request.param]


@pytest.fixture(
    scope="module",
    params=["unit", "diagonal1", "diagonal2", "cholesky", "dense"],
)
def metric(request):
    random = np.random.default_rng(5968)
    L = np.tril(random.standard_normal((5, 5)))
    L[np.diag_indices_from(L)] = np.exp(
        L[np.diag_indices_from(L)]
    ) + random.uniform(1.5, 2.5, len(L))
    return {
        "unit": (metrics.unit_metric, dict(metric=1.0, ndim=2)),
        "diagonal1": (
            metrics.diagonal_metric(0.5),
            dict(metric=0.5 ** 2, ndim=3),
        ),
        "diagonal2": (
            metrics.diagonal_metric(np.array([0.5, 0.3])),
            dict(metric=[0.5 ** 2, 0.3 ** 2], ndim=2),
        ),
        "cholesky": (
            metrics.cholesky_metric(L, lower=True),
            dict(metric=L @ L.T, ndim=len(L)),
        ),
        "dense": (
            metrics.dense_metric(L @ L.T),
            dict(metric=L @ L.T, ndim=len(L)),
        ),
    }[request.param]


@pytest.fixture(
    scope="module",
    params=["Exp", "ExpSquared", "Matern32", "Matern52", "RationalQuadratic"],
)
def metric_kernel(request, metric):
    tiny_metric, george_metric_args = metric
    return {
        "Exp": (
            kernels.Exp(tiny_metric),
            george.kernels.ExpKernel(**george_metric_args),
        ),
        "ExpSquared": (
            kernels.ExpSquared(tiny_metric),
            george.kernels.ExpSquaredKernel(**george_metric_args),
        ),
        "Matern32": (
            kernels.Matern32(tiny_metric),
            george.kernels.Matern32Kernel(**george_metric_args),
        ),
        "Matern52": (
            kernels.Matern52(tiny_metric),
            george.kernels.Matern52Kernel(**george_metric_args),
        ),
        "RationalQuadratic": (
            kernels.RationalQuadratic(tiny_metric, alpha=1.5),
            george.kernels.RationalQuadraticKernel(
                log_alpha=np.log(1.5), **george_metric_args
            ),
        ),
    }[request.param]


def compare_kernel_value(random, tiny_kernel, george_kernel):
    x1 = np.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    x2 = np.sort(random.uniform(0, 10, (45, george_kernel.ndim)))
    np.testing.assert_allclose(
        tiny_kernel.evaluate(x1, x2),
        george_kernel.get_value(x1, x2),
    )
    np.testing.assert_allclose(
        tiny_kernel.evaluate(x1, x1),
        george_kernel.get_value(x1),
    )
    np.testing.assert_allclose(
        tiny_kernel.evaluate_diag(x1),
        george_kernel.get_value(x1, diag=True),
    )


def compare_gps(random, tiny_kernel, george_kernel):
    x = np.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    t = np.sort(random.uniform(0, 10, (12, george_kernel.ndim)))
    y = np.sin(x[:, 0])
    diag = random.uniform(0.01, 0.1, 50)

    # Set up the GPs
    george_gp = george.GP(george_kernel)
    george_gp.compute(x, np.sqrt(diag))
    tiny_gp = GaussianProcess(tiny_kernel, x, diag=diag)

    # Likelihood
    np.testing.assert_allclose(
        tiny_gp.condition(y), george_gp.log_likelihood(y)
    )

    # Filtering
    np.testing.assert_allclose(
        tiny_gp.predict(),
        george_gp.predict(y, x, return_var=False, return_cov=False),
    )

    # Filtering with explicit value
    np.testing.assert_allclose(
        tiny_gp.predict(x),
        george_gp.predict(y, x, return_var=False, return_cov=False),
    )
    np.testing.assert_allclose(
        tiny_gp.predict(t),
        george_gp.predict(y, t, return_var=False, return_cov=False),
    )

    # Variance
    np.testing.assert_allclose(
        tiny_gp.predict(return_var=True)[1],
        george_gp.predict(y, x, return_var=True, return_cov=False)[1],
    )
    np.testing.assert_allclose(
        tiny_gp.predict(t, return_var=True)[1],
        george_gp.predict(y, t, return_var=True, return_cov=False)[1],
    )

    # Covariance
    np.testing.assert_allclose(
        tiny_gp.predict(return_cov=True)[1],
        george_gp.predict(y, x, return_var=False, return_cov=True)[1],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        tiny_gp.predict(t, return_cov=True)[1],
        george_gp.predict(y, t, return_var=False, return_cov=True)[1],
        rtol=1e-5,
    )


def test_kernel_value(random, kernel):
    tiny_kernel, george_kernel = kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)

    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_metric(metric):
    tiny_metric, george_metric_args = metric
    george_metric = george.metrics.Metric(**george_metric_args)
    for n in range(george_metric.ndim):
        e = np.zeros(george_metric.ndim)
        e[n] = 1.0
        np.testing.assert_allclose(
            tiny_metric(e), e.T @ np.linalg.solve(george_metric.to_matrix(), e)
        )


def test_metric_kernel_value(random, metric_kernel):
    tiny_kernel, george_kernel = metric_kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)

    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_cosine_kernel_value(random):
    tiny_kernel = kernels.Cosine(2.3)
    george_kernel = george.kernels.CosineKernel(log_period=np.log(2.3))
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_gp(random, kernel):
    tiny_kernel, george_kernel = kernel
    compare_gps(random, tiny_kernel, george_kernel)


def test_metric_gp(random, metric_kernel):
    tiny_kernel, george_kernel = metric_kernel
    compare_gps(random, tiny_kernel, george_kernel)


def test_cosine_gp(random):
    tiny_kernel = kernels.Cosine(2.3)
    george_kernel = george.kernels.CosineKernel(log_period=np.log(2.3))
    compare_gps(random, tiny_kernel, george_kernel)
