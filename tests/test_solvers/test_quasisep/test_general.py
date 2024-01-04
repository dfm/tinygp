# mypy: ignore-errors

import jax.numpy as jnp
from numpy import random as np_random

from tinygp.kernels import quasisep
from tinygp.test_utils import assert_allclose


def test_matmul():
    random = np_random.default_rng(1234)
    x1_ = jnp.sort(random.uniform(0, 10, 100))
    x2_ = jnp.sort(random.uniform(2, 8, 75))
    kernel = quasisep.Matern52(sigma=1.5, scale=3.4)

    for x1, x2 in [(x1_, x2_), (x1_, x1_), (x2_, x1_)]:
        y = jnp.sin(x2)[:, None]
        K = kernel(x1, x2)
        mat = kernel.to_general_qsm(x1, x2)
        assert_allclose(mat.matmul(y), K @ y)
