# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag

from tinygp.solvers.quasisep.block import Block
from tinygp.test_utils import assert_allclose


@pytest.fixture(params=[(10,), (10, 3), (4, 10, 3), (2, 4, 10, 3)])
def block_params(request):
    shape = request.param
    random = np.random.default_rng(1234)
    block_sizes = [1, 3, 2, 4]
    block = Block(*(random.uniform(size=(s, s)) for s in block_sizes))
    x = random.uniform(size=shape)
    return block, x


def test_block(block_params):
    block, x = block_params
    xt = x if len(x.shape) == 1 else jnp.swapaxes(x, -1, -2)
    block_ = block_diag(*block.blocks)
    assert len(block) == len(block_)
    assert block.shape == block_.shape
    assert_allclose(block @ x, block_ @ x)
    assert_allclose(xt @ block, xt @ block_)
    assert_allclose(block.T @ x, block_.T @ x)
    assert_allclose(xt @ block.T, xt @ block_.T)
