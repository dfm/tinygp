"""
While the usual definition of quasiseparable matrices is restricted to square
matrices, it is useful for our purposes to also implement some algorithms for a
somewhat more general class of rectangular quasiseparable matrices. These appear
in the calculations for the conditional Gaussian Process when interpolating and
extrapolating. We have not (yet?) worked through some of the more general
operations (like scalable matrix multiplies), but those may be possible to
derive.

This generalization isn't published anywhere as far as I know (please tell me if
there is a reference that you know of!), and maybe someday I'll come up with
notation that I'm satisfied with and try to write it up.
"""

from __future__ import annotations

__all__ = ["GeneralQSM"]

from functools import wraps
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from tinygp.helpers import JAXArray


def handle_matvec_shapes(
    func: Callable[[Any, JAXArray], JAXArray]
) -> Callable[[Any, JAXArray], JAXArray]:
    @wraps(func)
    def wrapped(self: Any, x: JAXArray) -> JAXArray:
        output_shape = (-1,) + x.shape[1:]
        result = func(self, jnp.reshape(x, (x.shape[0], -1)))
        return jnp.reshape(result, output_shape)

    return wrapped


class GeneralQSM(eqx.Module):
    """A rectangular ``(n1,n2)`` quasiseparable matrix with order ``m``

    Args:
        pl (n1, m): The lower left quasiseparable vectors.
        ql (n2, m): The lower right quasiseparable vectors.
        pu (n2, m): The upper right quasiseparable vectors.
        qu (n1, m): The upper left quasiseparable vectors.
        a (n1, m, m): The transition matrices.
        idx (n1,): The indices of the diagonal.
    """

    pl: JAXArray
    ql: JAXArray
    pu: JAXArray
    qu: JAXArray
    a: JAXArray
    idx: JAXArray

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix"""
        return (self.pl.shape[0], self.ql.shape[0])

    @jax.jit
    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        """The dot product of this matrix with a dense vector or matrix

        Args:
            x (n2, ...): A matrix or vector with leading dimension matching this
                matrix.
        """

        # Use a forward pass to dot the "lower" matrix
        def forward(f, data):  # type: ignore
            q, a, x = data
            fn = a @ f + jnp.outer(q, x)
            return fn, fn

        init = jnp.zeros_like(jnp.outer(self.ql[0], x[0]))
        _, f = jax.lax.scan(forward, init, (self.ql, self.a, x))
        idx = jnp.clip(self.idx, 0, f.shape[0] - 1)
        mask = jnp.logical_and(self.idx >= 0, self.idx < f.shape[0])
        lower = jax.vmap(jnp.dot)(jnp.where(mask[:, None], self.pl, 0), f[idx])

        # Then a backward pass to apply the "upper" matrix
        def backward(f, data):  # type: ignore
            p, a, x = data
            fn = a.T @ f + jnp.outer(p, x)
            return fn, fn

        init = jnp.zeros_like(jnp.outer(self.pu[-1], x[-1]))
        _, f = jax.lax.scan(
            backward,
            init,
            (self.pu, jnp.roll(self.a, -1, axis=0), x),
            reverse=True,
        )
        idx = jnp.clip(self.idx + 1, 0, f.shape[0] - 1)
        mask = jnp.logical_and(self.idx >= -1, self.idx < f.shape[0] - 1)
        upper = jax.vmap(jnp.dot)(jnp.where(mask[:, None], self.qu, 0), f[idx])

        return lower + upper

    def __matmul__(self, other: Any) -> Any:
        return self.matmul(other)
