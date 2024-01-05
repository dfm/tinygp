"""
The algorithms implemented in this subpackage are are mostly based on `Eidelman
& Gohberg (1999) <https://link.springer.com/article/10.1007%2FBF01300581>`_ and
`Foreman-Mackey et al. (2017) <https://arxiv.org/abs/1703.09710>`_.
"""

from __future__ import annotations

__all__ = [
    "DiagQSM",
    "StrictLowerTriQSM",
    "StrictUpperTriQSM",
    "LowerTriQSM",
    "UpperTriQSM",
    "SquareQSM",
    "SymmQSM",
]

import dataclasses
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray


def handle_matvec_shapes(
    func: Callable[[Any, JAXArray], JAXArray]
) -> Callable[[Any, JAXArray], JAXArray]:
    @wraps(func)
    def wrapped(self: Any, x: JAXArray) -> JAXArray:
        output_shape = x.shape
        result = func(self, jnp.reshape(x, (output_shape[0], -1)))
        return jnp.reshape(result, output_shape)

    return wrapped


class QSM(eqx.Module):
    """The base class for all square quasiseparable matrices

    This class has blanket implementations of the standard operations that are
    implemented for all QSMs, like addtion, subtraction, multiplication, and
    matrix multiplication.
    """

    # Must be higher than jax's
    __array_priority__ = 2000

    @abstractmethod
    def transpose(self) -> Any:
        """The matrix transpose as a QSM"""
        raise NotImplementedError

    @abstractmethod
    def matmul(self, x: JAXArray) -> JAXArray:
        """The dot product of this matrix with a dense vector or matrix

        Args:
            x (n, ...): A matrix or vector with leading dimension matching this
                matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def scale(self, other: JAXArray) -> QSM:
        """The multiplication of this matrix times a scalar, as a QSM"""
        raise NotImplementedError

    @property
    def T(self) -> Any:
        return self.transpose()

    def to_dense(self) -> JAXArray:
        """Render this representation to a dense matrix

        This implementation is not optimized and should really only ever be used
        for testing purposes.
        """
        return self.matmul(jnp.eye(self.shape[0]))

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix"""
        n = self.diag.shape[0]  # type: ignore
        return (n, n)

    def __iter__(self):  # type: ignore
        return (getattr(self, f.name) for f in dataclasses.fields(self))

    def __sub__(self, other: Any) -> Any:
        return self.__add__(-other)

    def __add__(self, other: Any) -> Any:
        from tinygp.solvers.quasisep.ops import elementwise_add

        return elementwise_add(self, other)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, QSM):
            from tinygp.solvers.quasisep.ops import elementwise_mul

            return elementwise_mul(self, other)
        else:
            assert jnp.ndim(other) <= 1
            return self.scale(other)

    def __rmul__(self, other: Any) -> Any:
        assert not isinstance(other, QSM)
        assert jnp.ndim(other) <= 1
        return self.scale(other)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, QSM):
            from tinygp.solvers.quasisep.ops import qsm_mul

            return qsm_mul(self, other)
        else:
            return self.matmul(other)

    def __rmatmul__(self, other: Any) -> Any:
        assert not isinstance(other, QSM)
        return (self.transpose() @ other.transpose()).transpose()


class DiagQSM(QSM):
    """A diagonal quasiseparable matrix

    Args:
        d (n,): The diagonal entries of the matrix as a 1-D array.
    """

    d: JAXArray

    @property
    def shape(self) -> tuple[int, int]:
        n = self.d.shape[0]
        return (n, n)

    def transpose(self) -> DiagQSM:
        return self

    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.d[:, None] * x

    def scale(self, other: JAXArray) -> DiagQSM:
        return DiagQSM(d=self.d * other)

    def self_add(self, other: DiagQSM) -> DiagQSM:
        """The sum of two :class:`DiagQSM` matrices"""
        return DiagQSM(d=self.d + other.d)

    def self_mul(self, other: DiagQSM) -> DiagQSM:
        """The elementwise product of two :class:`DiagQSM` matrices"""
        return DiagQSM(d=self.d * other.d)

    def __neg__(self) -> DiagQSM:
        return DiagQSM(d=-self.d)


class StrictLowerTriQSM(QSM):
    """A strictly lower triangular order ``m`` quasiseparable matrix

    Args:
        p (n, m): The left quasiseparable elements.
        q (n, m): The right quasiseparable elements.
        a (n, m, m): The transition matrices.
    """

    p: JAXArray
    q: JAXArray
    a: JAXArray

    @property
    def shape(self) -> tuple[int, int]:
        n = self.p.shape[0]
        return (n, n)

    def transpose(self) -> StrictUpperTriQSM:
        return StrictUpperTriQSM(p=self.p, q=self.q, a=self.a)

    @jax.jit
    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            q, a, x = data
            return a @ f + jnp.outer(q, x), f

        init = jnp.zeros_like(jnp.outer(self.q[0], x[0]))
        _, f = jax.lax.scan(impl, init, (self.q, self.a, x))
        return jax.vmap(jnp.dot)(self.p, f)

    def scale(self, other: JAXArray) -> StrictLowerTriQSM:
        return StrictLowerTriQSM(p=self.p * other, q=self.q, a=self.a)

    def self_add(self, other: StrictLowerTriQSM) -> StrictLowerTriQSM:
        """The sum of two :class:`StrictLowerTriQSM` matrices"""

        @jax.vmap
        def impl(
            self: StrictLowerTriQSM, other: StrictLowerTriQSM
        ) -> StrictLowerTriQSM:
            p1, q1, a1 = self
            p2, q2, a2 = other
            return StrictLowerTriQSM(
                p=jnp.concatenate((p1, p2)),
                q=jnp.concatenate((q1, q2)),
                a=block_diag(a1, a2),
            )

        return impl(self, other)

    def self_mul(self, other: StrictLowerTriQSM) -> StrictLowerTriQSM:
        """The elementwise product of two :class:`StrictLowerTriQSM` matrices"""
        i, j = np.meshgrid(np.arange(self.p.shape[1]), np.arange(other.p.shape[1]))
        i = i.flatten()
        j = j.flatten()
        return StrictLowerTriQSM(
            p=self.p[:, i] * other.p[:, j],
            q=self.q[:, i] * other.q[:, j],
            a=self.a[:, i[:, None], i[None, :]] * other.a[:, j[:, None], j[None, :]],
        )

    def __neg__(self) -> StrictLowerTriQSM:
        return StrictLowerTriQSM(p=-self.p, q=self.q, a=self.a)


class StrictUpperTriQSM(QSM):
    """A strictly upper triangular order ``m`` quasiseparable matrix

    The notation here is somewhat different from that in `Eidelman & Gohberg
    (1999) <https://link.springer.com/article/10.1007%2FBF01300581>`_, because
    we wanted to map ``StrictLowerTriQSM.transpose() -> StrictUpperTriQSM``
    while retaining the same names for each component. Therefore, our ``p`` is
    their ``h``, and our ``a`` is their ``b.T``.

    Args:
        p (n, m): The right quasiseparable elements.
        q (n, m): The left quasiseparable elements.
        a (n, m, m): The transition matrices.
    """

    p: JAXArray
    q: JAXArray
    a: JAXArray

    @property
    def shape(self) -> tuple[int, int]:
        n = self.p.shape[0]
        return (n, n)

    def transpose(self) -> StrictLowerTriQSM:
        return StrictLowerTriQSM(p=self.p, q=self.q, a=self.a)

    @jax.jit
    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        def impl(f, data):  # type: ignore
            p, a, x = data
            return a.T @ f + jnp.outer(p, x), f

        init = jnp.zeros_like(jnp.outer(self.p[-1], x[-1]))
        _, f = jax.lax.scan(impl, init, (self.p, self.a, x), reverse=True)
        return jax.vmap(jnp.dot)(self.q, f)

    def scale(self, other: JAXArray) -> StrictUpperTriQSM:
        return StrictUpperTriQSM(p=self.p, q=self.q * other, a=self.a)

    def self_add(self, other: StrictUpperTriQSM) -> StrictUpperTriQSM:
        """The sum of two :class:`StrictUpperTriQSM` matrices"""
        return self.transpose().self_add(other.transpose()).transpose()

    def self_mul(self, other: StrictUpperTriQSM) -> StrictUpperTriQSM:
        """The elementwise product of two :class:`StrictUpperTriQSM` matrices"""
        return self.transpose().self_mul(other.transpose()).transpose()

    def __neg__(self) -> StrictUpperTriQSM:
        return StrictUpperTriQSM(p=-self.p, q=self.q, a=self.a)


class LowerTriQSM(QSM):
    """A lower triangular quasiseparable matrix

    Args:
        diag: The diagonal elements.
        lower: The strictly lower triangular elements.
    """

    diag: DiagQSM
    lower: StrictLowerTriQSM

    def transpose(self) -> UpperTriQSM:
        return UpperTriQSM(diag=self.diag, upper=self.lower.transpose())

    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.diag.matmul(x) + self.lower.matmul(x)

    def scale(self, other: JAXArray) -> LowerTriQSM:
        return LowerTriQSM(diag=self.diag.scale(other), lower=self.lower.scale(other))

    def inv(self) -> LowerTriQSM:
        (d,) = self.diag
        p, q, a = self.lower
        g = 1 / d
        u = -g[:, None] * p
        v = g[:, None] * q
        b = a - jax.vmap(jnp.outer)(v, p)
        return LowerTriQSM(diag=DiagQSM(g), lower=StrictLowerTriQSM(p=u, q=v, a=b))

    @jax.jit
    @handle_matvec_shapes
    def solve(self, y: JAXArray) -> JAXArray:
        """Solve a linear system with this matrix

        If this matrix is called ``L``, this solves ``L @ x = y`` for ``x``
        given ``y``, using forward substitution.

        Args:
            y (n, ...): A matrix or vector with leading dimension matching this
                matrix.
        """

        def impl(fn, data):  # type: ignore
            ((cn,), (pn, wn, an)), yn = data
            xn = (yn - pn @ fn) / cn
            return an @ fn + jnp.outer(wn, xn), xn

        init = jnp.zeros_like(jnp.outer(self.lower.q[0], y[0]))
        _, x = jax.lax.scan(impl, init, (self, y))
        return x

    def __neg__(self) -> LowerTriQSM:
        return LowerTriQSM(diag=-self.diag, lower=-self.lower)


class UpperTriQSM(QSM):
    """A upper triangular quasiseparable matrix

    Args:
        diag: The diagonal elements.
        upper: The strictly upper triangular elements.
    """

    diag: DiagQSM
    upper: StrictUpperTriQSM

    def transpose(self) -> LowerTriQSM:
        return LowerTriQSM(diag=self.diag, lower=self.upper.transpose())

    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.diag.matmul(x) + self.upper.matmul(x)

    def scale(self, other: JAXArray) -> UpperTriQSM:
        return UpperTriQSM(diag=self.diag.scale(other), upper=self.upper.scale(other))

    def inv(self) -> UpperTriQSM:
        return self.transpose().inv().transpose()

    @jax.jit
    @handle_matvec_shapes
    def solve(self, y: JAXArray) -> JAXArray:
        """Solve a linear system with this matrix

        If this matrix is called ``U``, this solves ``U @ x = y`` for ``x``
        given ``y``, using backward substitution.

        Args:
            y (n, ...): A matrix or vector with leading dimension matching this
                matrix.
        """

        def impl(fn, data):  # type: ignore
            ((cn,), (pn, wn, an)), yn = data
            xn = (yn - wn @ fn) / cn
            return an.T @ fn + jnp.outer(pn, xn), xn

        init = jnp.zeros_like(jnp.outer(self.upper.p[-1], y[-1]))
        _, x = jax.lax.scan(impl, init, (self, y), reverse=True)
        return x

    def __neg__(self) -> UpperTriQSM:
        return UpperTriQSM(diag=-self.diag, upper=-self.upper)


class SquareQSM(QSM):
    """A general square order ``(m1, m2)`` quasiseparable matrix

    Args:
        diag: The diagonal elements.
        lower: The strictly lower triangular elements with order ``m1``.
        upper: The strictly upper triangular elements with order ``m2``.
    """

    diag: DiagQSM
    lower: StrictLowerTriQSM
    upper: StrictUpperTriQSM

    def transpose(self) -> SquareQSM:
        return SquareQSM(
            diag=self.diag,
            lower=self.upper.transpose(),
            upper=self.lower.transpose(),
        )

    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return self.diag.matmul(x) + self.lower.matmul(x) + self.upper.matmul(x)

    def scale(self, other: JAXArray) -> SquareQSM:
        return SquareQSM(
            diag=self.diag.scale(other),
            lower=self.lower.scale(other),
            upper=self.upper.scale(other),
        )

    def gram(self) -> SymmQSM:
        """The inner product of this matrix with itself

        If this matrix is called ``A``, the Gram matrix is ``A.T @ A``, and
        that's what this method computes. The result is a :class:`SymmQSM`.
        """
        # We know that this must result in symmetric matrix, but that won't be
        # enforced; we make it so! It might be possible to make this more
        # efficient, but perhaps jax is clever enough?
        M = self.transpose() @ self
        return SymmQSM(diag=M.diag, lower=M.lower)

    @jax.jit
    def inv(self) -> SquareQSM:
        """The inverse of this matrix"""
        (d,) = self.diag
        p, q, a = self.lower
        h, g, b = self.upper

        def forward(carry, data):  # type: ignore
            f = carry
            dk, pk, qk, ak, gk, hk, bk = data
            fhk = f @ hk
            fbk = f @ bk.T
            left = qk - ak @ fhk
            right = gk - pk @ fbk
            igk = 1 / (dk - pk @ fhk)
            sk = igk * left
            ellk = ak - jnp.outer(sk, pk)
            vk = igk * right
            delk = bk - jnp.outer(vk, hk)
            fk = ak @ fbk + igk * jnp.outer(left, right)
            return fk, (igk, sk, ellk, vk, delk)

        init = jnp.zeros_like(jnp.outer(q[0], g[0]))
        ig, s, ell, v, del_ = jax.lax.scan(forward, init, (d, p, q, a, g, h, b))[1]

        def backward(carry, data):  # type: ignore
            z = carry
            igk, pk, ak, hk, bk, sk, vk = data
            zsk = z @ sk
            zak = z @ ak
            lk = igk + vk @ zsk
            tk = vk @ zak - lk * pk
            uk = bk.T @ zsk - lk * hk
            zk = bk.T @ zak - jnp.outer(uk + lk * hk, pk) - jnp.outer(hk, tk)
            return zk, (lk, tk, uk)

        init = jnp.zeros_like(jnp.outer(h[-1], p[-1]))
        args = (ig, p, a, h, b, s, v)
        lam, t, u = jax.lax.scan(backward, init, args, reverse=True)[1]
        return SquareQSM(
            diag=DiagQSM(d=lam),
            lower=StrictLowerTriQSM(p=t, q=s, a=ell),
            upper=StrictUpperTriQSM(p=u, q=v, a=del_),
        )

    def __neg__(self) -> SquareQSM:
        return SquareQSM(diag=-self.diag, lower=-self.lower, upper=-self.upper)


class SymmQSM(QSM):
    """A symmetric order ``m`` quasiseparable matrix

    Args:
        diag: The diagonal elements.
        lower: The strictly lower triangular elements with order ``m``.
    """

    diag: DiagQSM
    lower: StrictLowerTriQSM

    def transpose(self) -> SymmQSM:
        return self

    @handle_matvec_shapes
    def matmul(self, x: JAXArray) -> JAXArray:
        return (
            self.diag.matmul(x)
            + self.lower.matmul(x)
            + self.lower.transpose().matmul(x)
        )

    def scale(self, other: JAXArray) -> SymmQSM:
        return SymmQSM(diag=self.diag.scale(other), lower=self.lower.scale(other))

    @jax.jit
    def inv(self) -> SymmQSM:
        """The inverse of this matrix"""
        (d,) = self.diag
        p, q, a = self.lower

        def forward(carry, data):  # type: ignore
            f = carry
            dk, pk, qk, ak = data
            fpk = f @ pk
            left = qk - ak @ fpk
            igk = 1 / (dk - pk @ fpk)
            sk = igk * left
            ellk = ak - jnp.outer(sk, pk)
            fk = ak @ f @ ak.T + igk * jnp.outer(left, left.T)
            return fk, (igk, sk, ellk)

        init = jnp.zeros_like(jnp.outer(q[0], q[0]))
        ig, s, ell = jax.lax.scan(forward, init, (d, p, q, a))[1]

        def backward(carry, data):  # type: ignore
            z = carry
            igk, pk, ak, sk = data
            zak = z @ ak
            skzak = sk @ zak
            lk = igk + sk @ z @ sk
            tk = skzak - lk * pk
            zk = ak.T @ zak - jnp.outer(skzak, pk) - jnp.outer(pk, tk)
            return zk, (lk, tk)

        init = jnp.zeros_like(jnp.outer(p[-1], p[-1]))
        lam, t = jax.lax.scan(backward, init, (ig, p, a, s), reverse=True)[1]
        return SymmQSM(diag=DiagQSM(d=lam), lower=StrictLowerTriQSM(p=t, q=s, a=ell))

    @jax.jit
    def cholesky(self) -> LowerTriQSM:
        """The Cholesky decomposition of this matrix

        If this matrix is called ``A``, this method returns the
        :class:`LowerTriQSM` ``L`` such that ``L @ L.T = A``.
        """
        (d,) = self.diag
        p, q, a = self.lower

        def impl(carry, data):  # type: ignore
            fp = carry
            dk, pk, qk, ak = data
            ck = jnp.sqrt(dk - pk @ fp @ pk)
            tmp = fp @ ak.T
            wk = (qk - pk @ tmp) / ck
            fk = ak @ tmp + jnp.outer(wk, wk)
            return fk, (ck, wk)

        init = jnp.zeros_like(jnp.outer(q[0], q[0]))
        _, (c, w) = jax.lax.scan(impl, init, (d, p, q, a))
        return LowerTriQSM(diag=DiagQSM(c), lower=StrictLowerTriQSM(p=p, q=w, a=a))

    def __neg__(self) -> SymmQSM:
        return SymmQSM(diag=-self.diag, lower=-self.lower)
