# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinygp import GaussianProcess, kernels, transforms


def test_affine():
    kernel0 = kernels.Matern32(4.5)
    kernel1 = transforms.Affine(4.5, kernels.Matern32())
    kernel2 = transforms.Affine(4.5 ** 2, kernels.Matern32(), variance=True)
    np.testing.assert_allclose(
        kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
    )
    np.testing.assert_allclose(
        kernel0.evaluate(0.5, 0.1), kernel2.evaluate(0.5, 0.1)
    )


def test_subspace():
    kernel = transforms.Subspace(1, kernels.Matern32())
    np.testing.assert_allclose(
        kernel.evaluate(np.array([0.5, 0.1]), np.array([-0.4, 0.7])),
        kernel.evaluate(np.array([100.5, 0.1]), np.array([-70.4, 0.7])),
    )
