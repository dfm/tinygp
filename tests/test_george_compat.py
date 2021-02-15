# -*- coding: utf-8 -*-

import numpy as np
import pytest
from jax.config import config

from tinygp import kernels, metrics

george = pytest.importorskip("george")

config.update("jax_enable_x64", True)


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


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


def test_metric(metric):
    tiny_metric, george_metric_args = metric
    george_metric = george.metrics.Metric(**george_metric_args)
    for n in range(george_metric.ndim):
        e = np.zeros(george_metric.ndim)
        e[n] = 1.0
        np.testing.assert_allclose(
            tiny_metric(e), e.T @ np.linalg.solve(george_metric.to_matrix(), e)
        )


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


def test_metric_kernel_value(random, metric_kernel):
    tiny_kernel, george_kernel = metric_kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)

    # What about a scaled version
    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_cosine_kernel_value(random):
    tiny_kernel = kernels.Cosine(2.3)
    george_kernel = george.kernels.CosineKernel(log_period=np.log(2.3))
    compare_kernel_value(random, tiny_kernel, george_kernel)
