# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.kernels import quasisep


def test_matmul():
    random = np.random.default_rng(1234)
    x1 = np.sort(random.uniform(0, 10, 100))
    x2 = np.sort(random.uniform(2, 8, 75))
    kernel = quasisep.Matern52(sigma=1.5, scale=3.4)

    for (x1, x2) in [(x1, x2), (x1, x1), (x2, x1)]:
        y = np.sin(x2)[:, None]
        K = kernel(x1, x2)
        mat = kernel.to_general_qsm(x1, x2)
        np.testing.assert_allclose(mat.matmul(y), K @ y)
