from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray


def ensure_dense(x: Any) -> Any:
    """Convert a Block to a dense array, passing through non-Block inputs."""
    if isinstance(x, Block):
        return x.to_dense()
    return x


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

    @property
    def mT(self) -> "Block":
        return Block(*(jnp.swapaxes(b, -1, -2) for b in self.blocks))

    def to_dense(self) -> JAXArray:
        ndim = self.ndim
        assert ndim >= 2
        if ndim == 2:
            return block_diag(*self.blocks)
        return jax.vmap(lambda *bs: Block(*bs).to_dense())(*self.blocks)

    @jax.jit
    def __mul__(self, other: Any) -> "Block":
        return Block(*(b * other for b in self.blocks))

    def __rmul__(self, other: Any) -> "Block":
        return self.__mul__(other)

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
                for b1, b2 in zip(self.blocks, other.blocks, strict=True)
            )
            return Block(
                *(b1 @ b2 for b1, b2 in zip(self.blocks, other.blocks, strict=True))
            )
        ndim = np.ndim(other)
        assert ndim >= 1
        idx = 0
        ys = []
        for b in self.blocks:
            size = np.shape(b)[-1]
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
        idx = 0
        ys = []
        for b in self.blocks:
            size = np.shape(b)[-2]
            x = other[..., idx : idx + size]
            ys.append(x @ b)
            idx += size
        return jnp.concatenate(ys, axis=-1)
