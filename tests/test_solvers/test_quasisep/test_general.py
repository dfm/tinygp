# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.solvers.quasisep import kernels
from tinygp.solvers.quasisep.general import GeneralQSM


def test_matmul():
    random = np.random.default_rng(1234)
    x1 = np.sort(random.uniform(0, 10, 100))
    x2 = np.sort(random.uniform(2, 8, 75))
    kernel = kernels.Matern52(sigma=1.5, scale=3.4)

    for (x1, x2) in [(x1, x2), (x1, x1), (x2, x1)]:
        y = np.sin(x2)[:, None]
        K = kernel(x1, x2)

        idx = jnp.searchsorted(x2, x1, side="right") - 1
        a = jax.vmap(kernel.A)(np.append(x2[0], x2[:-1]), x2)
        ql = jax.vmap(kernel.q)(x2)
        pl = jax.vmap(kernel.p)(x1)
        qu = jax.vmap(kernel.q)(x1)
        pu = jax.vmap(kernel.p)(x2)

        i = jnp.clip(idx, 0, x2.shape[0] - 1)
        pl = jax.vmap(jnp.dot)(pl, jax.vmap(kernel.A)(x2[i], x1))

        i = jnp.clip(idx + 1, 0, x2.shape[0] - 1)
        qu = jax.vmap(jnp.dot)(jax.vmap(kernel.A)(x1, x2[i]), qu)

        mat = GeneralQSM(pl=pl, ql=ql, pu=pu, qu=qu, a=a, idx=idx)
        np.testing.assert_allclose(mat.matmul(y), K @ y)
