from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray


class Block(eqx.Module):
    blocks: tuple[Any, ...]
    __array_priority__ = 1999

    def __init__(self, *blocks: Any):
        self.blocks = blocks

    def __getitem__(self, idx: Any) -> "Block":
        return Block(*(b[idx] for b in self.blocks))

    def __len__(self) -> int:
        assert all(np.ndim(b) == 2 for b in self.blocks)
        return sum(len(b) for b in self.blocks)

    @property
    def ndim(self) -> int:
        (ndim,) = {np.ndim(b) for b in self.blocks}
        return ndim

    @property
    def shape(self) -> tuple[int, int]:
        size = len(self)
        return (size, size)

    def transpose(self) -> "Block":
        return Block(*(b.transpose() for b in self.blocks))

    @property
    def T(self) -> "Block":
        return self.transpose()

    def to_dense(self) -> JAXArray:
        assert all(np.ndim(b) == 2 for b in self.blocks)
        return block_diag(*self.blocks)

    @jax.jit
    def __mul__(self, other: Any) -> "Block":
        return Block(*(b * other for b in self.blocks))

    @jax.jit
    def __add__(self, other: Any) -> Any:
        if isinstance(other, Block):
            assert len(self) == len(other)
            assert all(
                np.shape(b1) == np.shape(b2)
                for b1, b2 in zip(self.blocks, other.blocks)
            )
            return Block(*(b1 + b2 for b1, b2 in zip(self.blocks, other.blocks)))
        else:
            # TODO(dfm): This could be optimized to avoid converting to dense.
            return self.to_dense() + other

    def __radd__(self, other: Any) -> Any:
        # TODO(dfm): This could be optimized to avoid converting to dense.
        return other + self.to_dense()

    @jax.jit
    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Block):
            assert len(self) == len(other)
            assert all(
                np.shape(b1) == np.shape(b2)
                for b1, b2 in zip(self.blocks, other.blocks)
            )
            return Block(*(b1 - b2 for b1, b2 in zip(self.blocks, other.blocks)))
        else:
            # TODO(dfm): This could be optimized to avoid converting to dense.
            return self.to_dense() - other

    def __rsub__(self, other: Any) -> Any:
        # TODO(dfm): This could be optimized to avoid converting to dense.
        return other - self.to_dense()

    @jax.jit
    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, Block):
            assert len(self.blocks) == len(other.blocks)
            assert all(
                np.shape(b1) == np.shape(b2)
                for b1, b2 in zip(self.blocks, other.blocks)
            )
            return Block(*(b1 @ b2 for b1, b2 in zip(self.blocks, other.blocks)))
        assert all(np.ndim(b) == 2 for b in self.blocks)
        ndim = np.ndim(other)
        assert ndim >= 1
        idx = 0
        ys = []
        for b in self.blocks:
            size = len(b)
            x = (
                other[idx : idx + size]
                if ndim == 1
                else other[..., idx : idx + size, :]
            )
            ys.append(b @ x)
            idx += size
        return jnp.concatenate(ys) if ndim == 1 else jnp.concatenate(ys, axis=-2)

    @jax.jit
    def __rmatmul__(self, other: Any) -> Any:
        assert all(np.ndim(b) == 2 for b in self.blocks)
        idx = 0
        ys = []
        for b in self.blocks:
            size = len(b)
            x = other[..., idx : idx + size]
            ys.append(x @ b)
            idx += size
        return jnp.concatenate(ys, axis=-1)
