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
from collections.abc import Callable
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray
from tinygp.solvers.quasisep.block import ensure_dense


def _strict_lower_tri_matmul_scan(
    p: JAXArray, q: JAXArray, a: JAXArray, x: JAXArray
) -> JAXArray:
    """Sequential O(n) matmul using jax.lax.scan."""

    def impl(f, data):  # type: ignore
        q_n, a_n, x_n = data
        return a_n @ f + jnp.outer(q_n, x_n), f

    init = jnp.zeros_like(jnp.outer(q[0], x[0]))
    _, f = jax.lax.scan(impl, init, (q, a, x))
    return jax.vmap(jnp.dot)(p, f)


def _strict_lower_tri_matmul_associative_scan(
    p: JAXArray, q: JAXArray, a: JAXArray, x: JAXArray
) -> JAXArray:
    """Parallel O(n log n) matmul using jax.lax.associative_scan.

    The sequential recurrence is the affine map:

        f[0] = 0
        f[n] = a[n-1] @ f[n-1] + outer(q[n-1], x[n-1])

    Each step is an affine map (A_k, b_k): x -> A_k @ x + b_k, and the
    composition of two such maps is associative:

        (A_R, b_R) . (A_L, b_L) = (A_R @ A_L, A_R @ b_L + b_R)

    An inclusive prefix scan over these elements yields the composed map at each
    position. The b-component of result[n] equals f[n+1], so we shift right by
    one and prepend zeros to recover f[0..N-1].
    """

    def combine(left, right):  # type: ignore
        A_l, b_l = left
        A_r, b_r = right
        return (A_r @ A_l, A_r @ b_l + b_r)

    b = jax.vmap(jnp.outer)(q, x)  # (N, m, k)
    _, cumul_b = jax.lax.associative_scan(combine, (a, b))

    # cumul_b[n] = f[n+1], so shift right and prepend zeros
    f = jnp.concatenate([jnp.zeros_like(b[:1]), cumul_b[:-1]], axis=0)
    return jax.vmap(jnp.dot)(p, f)


# Default to the sequential implementation; swap to
# _strict_lower_tri_matmul_associative_scan for GPU parallelism.
_strict_lower_tri_matmul = _strict_lower_tri_matmul_scan


def _cholesky_scan(
    d: JAXArray, p: JAXArray, q: JAXArray, a: JAXArray
) -> tuple[JAXArray, JAXArray]:
    """Sequential O(n) Cholesky using jax.lax.scan."""

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
    return c, w


def _cholesky_associative_scan(
    d: JAXArray, p: JAXArray, q: JAXArray, a: JAXArray
) -> tuple[JAXArray, JAXArray]:
    """Parallel O(n log n) Cholesky using jax.lax.associative_scan.

    The sequential Cholesky involves a discrete Riccati recursion:

        F[0] = 0
        c[k] = sqrt(d[k] - p[k]^T F[k] p[k])
        w[k] = (q[k] - p[k]^T F[k] a[k]^T) / c[k]
        F[k+1] = a[k] F[k] a[k]^T + outer(w[k], w[k])

    This Riccati recursion can be represented as a Linear Fractional
    Transformation (LFT / Möbius transformation) on m×m matrices:

        F[k+1] = (M11[k] F[k] + M12[k]) (M21[k] F[k] + M22[k])^{-1}

    where M[k] is a 2m×2m matrix. LFTs compose by matrix multiplication,
    so a parallel prefix scan over M[k] gives the composed LFT at each
    position, from which F[k] can be extracted.

    For numerical stability, each scan element tracks both the LFT matrix
    M and the accumulated F value as a pair. The F component is computed
    via short-range LFT applications (at most O(log n) depth), avoiding
    the exponential condition number growth of long matrix products.
    """
    N = d.shape[0]
    m = q.shape[1]

    # --- Step 1: Compute the 2m×2m LFT matrix for each step via SVD ---
    Ms = _compute_cholesky_lft_matrices(d, p, q, a, m)

    # Normalize each M to unit Frobenius norm
    norms = jnp.linalg.norm(Ms.reshape(N, -1), axis=1)
    Ms = Ms / norms[:, None, None]

    # Initial F values: LFT(M_k, 0) = M12 @ inv(M22)
    def lft_at_zero(Mk):  # type: ignore
        M12 = Mk[:m, m:]
        M22 = Mk[m:, m:]
        return jnp.linalg.solve(M22.T, M12.T).T

    F_locals = jax.vmap(lft_at_zero)(Ms)

    # --- Step 2: Associative scan with (M, F) pair ---
    def combine(left, right):  # type: ignore
        M_left, F_left = left
        M_right, F_right = right

        # Apply LFT(M_right, F_left) for stable F propagation
        M11 = M_right[..., :m, :m]
        M12 = M_right[..., :m, m:]
        M21 = M_right[..., m:, :m]
        M22 = M_right[..., m:, m:]
        numer = M11 @ F_left + M12
        denom = M21 @ F_left + M22
        FpT = jnp.linalg.solve(
            jnp.swapaxes(denom, -2, -1), jnp.swapaxes(numer, -2, -1)
        )
        F_combined = jnp.swapaxes(FpT, -2, -1)

        # Compose M with normalization to prevent overflow
        M_combined = M_right @ M_left
        M_combined = M_combined / jnp.linalg.norm(
            M_combined, axis=(-2, -1), keepdims=True
        )

        return (M_combined, F_combined)

    _, Fs_scanned = jax.lax.associative_scan(combine, (Ms, F_locals))
    # Fs_scanned[k] = F_{k+1}, so shift right and prepend F_0 = 0
    F0 = jnp.zeros((m, m), dtype=d.dtype)
    Fs = jnp.concatenate([F0[None], Fs_scanned[:-1]], axis=0)

    # --- Step 3: Compute c[k] and w[k] from F[k] ---
    def compute_cw(dk, pk, qk, ak, Fk):  # type: ignore
        ck = jnp.sqrt(dk - pk @ Fk @ pk)
        tmp = Fk @ ak.T
        wk = (qk - pk @ tmp) / ck
        return ck, wk

    c, w = jax.vmap(compute_cw)(d, p, q, a, Fs)
    return c, w


def _compute_cholesky_lft_matrices(
    d: JAXArray, p: JAXArray, q: JAXArray, a: JAXArray, m: int
) -> JAXArray:
    """Compute 2m×2m LFT matrices for each Cholesky step via SVD.

    For each step k, finds M_k such that the Riccati step
    F' = f(F, d_k, p_k, q_k, a_k) equals the LFT
    F' = (M11 F + M12)(M21 F + M22)^{-1}.

    M_k is determined (up to scale) by evaluating the Riccati step at
    several probe matrices and finding the null space of the resulting
    linear system via SVD.
    """
    idx = jnp.arange(m, dtype=d.dtype)
    Im = jnp.eye(m, dtype=d.dtype)
    Imm = jnp.eye(m * m, dtype=d.dtype)

    # Deterministic probe matrices (symmetric, moderate scale)
    n_probes = 10

    def _make_probe(k):  # type: ignore
        B = jnp.sin(
            (k + 1.0) * (idx[:, None] * m + idx[None, :] + 1.0) * 0.73
        )
        return 0.05 * (B + B.T) / 2

    probes = jnp.stack([_make_probe(k) for k in range(n_probes)])

    def _compute_one_M(dk, pk, qk, ak):  # type: ignore
        def riccati_step(F):  # type: ignore
            ck = jnp.sqrt(dk - pk @ F @ pk)
            tmp = F @ ak.T
            wk = (qk - pk @ tmp) / ck
            return ak @ tmp + jnp.outer(wk, wk)

        Fps = jax.vmap(riccati_step)(probes)

        # Build linear system: for each probe (F, F'),
        # (I⊗F^T) vec(M11) + vec(M12) - (F'⊗F^T) vec(M21)
        #   - (F'⊗I) vec(M22) = 0
        def build_row_block(F, Fp):  # type: ignore
            return jnp.concatenate(
                [
                    jnp.kron(Im, F.T),
                    Imm,
                    -jnp.kron(Fp, F.T),
                    -jnp.kron(Fp, Im),
                ],
                axis=1,
            )

        rows = jax.vmap(build_row_block)(probes, Fps)
        system = rows.reshape(-1, 4 * m * m)

        _, S, Vt = jnp.linalg.svd(system, full_matrices=True)
        null_vec = Vt[-1]

        M11 = null_vec[0 * m * m : 1 * m * m].reshape(m, m)
        M12 = null_vec[1 * m * m : 2 * m * m].reshape(m, m)
        M21 = null_vec[2 * m * m : 3 * m * m].reshape(m, m)
        M22 = null_vec[3 * m * m : 4 * m * m].reshape(m, m)

        return jnp.block([[M11, M12], [M21, M22]])

    return jax.vmap(_compute_one_M)(d, p, q, a)


# Default to the sequential implementation; swap to
# _cholesky_associative_scan for GPU parallelism.
_cholesky = _cholesky_scan


def handle_matvec_shapes(
    func: Callable[[Any, JAXArray], JAXArray],
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
        return _strict_lower_tri_matmul(self.p, self.q, self.a, x)

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
                a=block_diag(ensure_dense(a1), ensure_dense(a2)),
            )

        return impl(self, other)

    def self_mul(self, other: StrictLowerTriQSM) -> StrictLowerTriQSM:
        """The elementwise product of two :class:`StrictLowerTriQSM` matrices"""
        # vmap is needed because a batched Block has 3D block arrays that
        # block_diag (used by to_dense) cannot handle without unbatching.
        self_a = jax.vmap(ensure_dense)(self.a)
        other_a = jax.vmap(ensure_dense)(other.a)
        i, j = np.meshgrid(np.arange(self.p.shape[1]), np.arange(other.p.shape[1]))
        i = i.flatten()
        j = j.flatten()
        return StrictLowerTriQSM(
            p=self.p[:, i] * other.p[:, j],
            q=self.q[:, i] * other.q[:, j],
            a=self_a[:, i[:, None], i[None, :]] * other_a[:, j[:, None], j[None, :]],
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
        c, w = _cholesky(d, p, q, a)
        return LowerTriQSM(diag=DiagQSM(c), lower=StrictLowerTriQSM(p=p, q=w, a=a))

    def __neg__(self) -> SymmQSM:
        return SymmQSM(diag=-self.diag, lower=-self.lower)
