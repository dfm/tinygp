from __future__ import annotations

__all__ = ["JAXArray", "dataclass", "field"]

from typing import Any

import equinox as eqx
import jax

JAXArray = jax.Array


# The following is just for backwards compatibility since tinygp used to provide a
# custom dataclass implementation
field = eqx.field


def dataclass(clz: type[Any]) -> type[Any]:
    return clz
