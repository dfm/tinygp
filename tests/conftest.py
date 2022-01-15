# -*- coding: utf-8 -*-
# mypy: ignore-errors

import numpy as np
import pytest

from jax.config import config

from tinygp import metrics

config.update("jax_enable_x64", True)


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
