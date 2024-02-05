from __future__ import annotations

__all__ = ["elementwise_add", "elementwise_mul", "qsm_mul"]

from typing import TypeVar

import jax
import jax.numpy as jnp

from tinygp.helpers import JAXArray
from tinygp.solvers.quasisep.core import (
    QSM,
    DiagQSM,
    LowerTriQSM,
    SquareQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    SymmQSM,
    UpperTriQSM,
)


@jax.jit
def elementwise_add(a: QSM, b: QSM) -> QSM | None:
    diag_a, lower_a, upper_a = deconstruct(a)
    diag_b, lower_b, upper_b = deconstruct(b)

    diag = add_two(diag_a, diag_b)
    lower = add_two(lower_a, lower_b)
    upper = add_two(upper_a, upper_b)

    is_symm_a = isinstance(a, (DiagQSM, SymmQSM))
    is_symm_b = isinstance(b, (DiagQSM, SymmQSM))
    return construct(diag, lower, upper, is_symm_a and is_symm_b)


@jax.jit
def elementwise_mul(a: QSM, b: QSM) -> QSM | None:
    diag_a, lower_a, upper_a = deconstruct(a)
    diag_b, lower_b, upper_b = deconstruct(b)

    diag = mul_two(diag_a, diag_b)
    lower = mul_two(lower_a, lower_b)
    upper = mul_two(upper_a, upper_b)

    is_symm_a = isinstance(a, (DiagQSM, SymmQSM))
    is_symm_b = isinstance(b, (DiagQSM, SymmQSM))
    return construct(diag, lower, upper, is_symm_a and is_symm_b)


@jax.jit
def qsm_mul(a: QSM, b: QSM) -> QSM | None:
    diag_a, lower_a, upper_a = deconstruct(a)
    diag_b, lower_b, upper_b = deconstruct(b)

    # Special case for the product of two diagonal matrices
    if lower_a is None and upper_a is None and lower_b is None and upper_b is None:
        assert diag_a is not None and diag_b is not None
        return DiagQSM(d=diag_a * diag_b)

    if lower_a is not None and upper_b is not None:

        def calc_phi(phi, data):  # type: ignore
            a, b, q, g = data
            return a @ phi @ b.T + jnp.outer(q, g), phi

        init = jnp.zeros_like(jnp.outer(lower_a.q[0], upper_b.q[0]))
        args = (lower_a.a, upper_b.a, lower_a.q, upper_b.q)
        _, phi = jax.lax.scan(calc_phi, init, args)

    else:
        phi = None

    if upper_a is not None and lower_b is not None:

        def calc_psi(psi, data):  # type: ignore
            a, b, q, g = data
            return a.T @ psi @ b + jnp.outer(q, g), psi

        init = jnp.zeros_like(jnp.outer(upper_a.q[-1], lower_b.p[-1]))
        args = (upper_a.a, lower_b.a, upper_a.p, lower_b.p)
        _, psi = jax.lax.scan(calc_psi, init, args, reverse=True)

    else:
        psi = None

    @jax.vmap
    def impl(
        diag_a: DiagQSM | None,
        lower_a: StrictLowerTriQSM | None,
        upper_a: StrictUpperTriQSM | None,
        diag_b: DiagQSM | None,
        lower_b: StrictLowerTriQSM | None,
        upper_b: StrictUpperTriQSM | None,
        phi: JAXArray | None,
        psi: JAXArray | None,
    ) -> tuple[
        DiagQSM | None,
        StrictLowerTriQSM | None,
        StrictUpperTriQSM | None,
    ]:
        # Note: the order of g and h is flipped vs the paper!

        alpha = None
        beta = None
        theta = None
        eta = None
        lam = None

        if diag_b is not None and lower_a is not None:
            alpha = lower_a.q * diag_b.d

        if diag_a is not None and lower_b is not None:
            beta = diag_a.d * lower_b.p

        if diag_a is not None and upper_b is not None:
            theta = diag_a.d * upper_b.q

        if diag_b is not None and upper_a is not None:
            eta = upper_a.p * diag_b.d

        if diag_a is not None and diag_b is not None:
            lam = diag_a.d * diag_b.d

        if lower_a is not None and phi is not None and upper_b is not None:
            alpha = none_safe_add(alpha, lower_a.a @ phi @ upper_b.p)
            theta = none_safe_add(theta, lower_a.p @ phi @ upper_b.a.transpose())
            lam = none_safe_add(lam, lower_a.p @ phi @ upper_b.p)

        if upper_a is not None and psi is not None and lower_b is not None:
            beta = none_safe_add(beta, upper_a.q @ psi @ lower_b.a)
            eta = none_safe_add(eta, upper_a.a.transpose() @ psi @ lower_b.q)
            lam = none_safe_add(lam, upper_a.q @ psi @ lower_b.q)

        s = [alpha] if alpha is not None else []
        s += [lower_b.q] if lower_b is not None else []

        t = [lower_a.p] if lower_a is not None else []
        t += [beta] if beta is not None else []

        v = [upper_a.q] if upper_a is not None else []
        v += [theta] if theta is not None else []

        u = [eta] if eta is not None else []
        u += [upper_b.p] if upper_b is not None else []

        if lower_a is not None and lower_b is not None:
            ell = jnp.concatenate(
                (
                    jnp.concatenate(
                        (lower_a.a, jnp.outer(lower_a.q, lower_b.p)), axis=-1
                    ),
                    jnp.concatenate(
                        (
                            jnp.zeros((lower_b.a.shape[0], lower_a.a.shape[0])),
                            lower_b.a,
                        ),
                        axis=-1,
                    ),
                ),
                axis=0,
            )
        else:
            ell = (
                lower_a.a
                if lower_a is not None
                else lower_b.a if lower_b is not None else None
            )

        if upper_a is not None and upper_b is not None:
            delta = jnp.concatenate(
                (
                    jnp.concatenate(
                        (
                            upper_a.a,
                            jnp.zeros((upper_a.a.shape[0], upper_b.a.shape[0])),
                        ),
                        axis=-1,
                    ),
                    jnp.concatenate(
                        (jnp.outer(upper_b.q, upper_a.p), upper_b.a), axis=-1
                    ),
                ),
                axis=0,
            )

        else:
            delta = (
                upper_a.a
                if upper_a is not None
                else upper_b.a if upper_b is not None else None
            )

        return (
            DiagQSM(d=lam) if lam is not None else None,
            (
                StrictLowerTriQSM(p=jnp.concatenate(t), q=jnp.concatenate(s), a=ell)
                if len(t) and len(s) and ell is not None
                else None
            ),
            (
                StrictUpperTriQSM(p=jnp.concatenate(u), q=jnp.concatenate(v), a=delta)
                if len(u) and len(v) and delta is not None
                else None
            ),
        )

    diag, lower, upper = impl(
        diag_a, lower_a, upper_a, diag_b, lower_b, upper_b, phi, psi
    )
    is_symm_a = isinstance(a, (DiagQSM, SymmQSM))
    is_symm_b = isinstance(b, (DiagQSM, SymmQSM))
    return construct(diag, lower, upper, is_symm_a and is_symm_b)


def deconstruct(
    a: QSM,
) -> tuple[DiagQSM | None, StrictLowerTriQSM | None, StrictUpperTriQSM | None]:
    diag = a if isinstance(a, DiagQSM) else getattr(a, "diag", None)
    lower = a if isinstance(a, StrictLowerTriQSM) else getattr(a, "lower", None)
    upper = None
    if isinstance(a, StrictUpperTriQSM):
        upper = a
    elif isinstance(a, SymmQSM):
        upper = a.lower.transpose()
    elif hasattr(a, "upper"):
        upper = a.upper  # type: ignore
    return diag, lower, upper


def construct(
    diag: DiagQSM | None,
    lower: StrictLowerTriQSM | None,
    upper: StrictUpperTriQSM | None,
    symm: bool,
) -> QSM | None:
    if lower is None and upper is None:
        return diag

    if symm:
        assert diag is not None
        assert lower is not None
        return SymmQSM(diag=diag, lower=lower)

    if lower is None and upper is None:
        return diag

    if lower is None:
        if diag is None:
            return upper
        else:
            assert upper is not None
            return UpperTriQSM(diag=diag, upper=upper)

    elif upper is None:
        if diag is None:
            return lower
        else:
            assert lower is not None
            return LowerTriQSM(diag=diag, lower=lower)

    elif diag is None:
        # We would hit here if we add a StrictLower to a StrictUpper; is this an
        # ok way to handle that?
        return None

    return SquareQSM(diag=diag, lower=lower, upper=upper)


F = TypeVar("F", DiagQSM, StrictLowerTriQSM, StrictUpperTriQSM)


def add_two(a: F | None, b: F | None) -> F | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a.self_add(b)  # type: ignore


def mul_two(a: F | None, b: F | None) -> F | None:
    if a is None or b is None:
        return None
    return a.self_mul(b)  # type: ignore


def none_safe_add(a: JAXArray | None, b: JAXArray | None) -> JAXArray | None:
    if a is not None and b is not None:
        return a + b
    return a if a is not None else b
