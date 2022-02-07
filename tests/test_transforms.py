# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np

from tinygp import kernels, transforms


def test_linear():
    kernel0 = kernels.Matern32(4.5)
    kernel1 = transforms.Linear(1 / 4.5, kernels.Matern32())
    np.testing.assert_allclose(
        kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
    )


def test_multivariate_linear():
    kernel0 = kernels.Matern32(4.5)
    kernel1 = transforms.Linear(jnp.full(3, 1 / 4.5), kernels.Matern32())
    np.testing.assert_allclose(
        kernel0.evaluate(jnp.full(3, 0.5), jnp.full(3, 0.1)),
        kernel1.evaluate(jnp.full(3, 0.5), jnp.full(3, 0.1)),
    )


def test_cholesky():
    kernel0 = kernels.Matern32(4.5)
    kernel1 = transforms.Cholesky(4.5, kernels.Matern32())
    np.testing.assert_allclose(
        kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
    )


def test_multivariate_cholesky():
    kernel0 = kernels.Matern32(4.5)
    kernel1 = transforms.Cholesky(jnp.full(3, 4.5), kernels.Matern32())
    np.testing.assert_allclose(
        kernel0.evaluate(jnp.full(3, 0.5), jnp.full(3, 0.1)),
        kernel1.evaluate(jnp.full(3, 0.5), jnp.full(3, 0.1)),
    )


def test_subspace():
    kernel = transforms.Subspace(1, kernels.Matern32())
    np.testing.assert_allclose(
        kernel.evaluate(np.array([0.5, 0.1]), np.array([-0.4, 0.7])),
        kernel.evaluate(np.array([100.5, 0.1]), np.array([-70.4, 0.7])),
    )
