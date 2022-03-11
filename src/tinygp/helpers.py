# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["JAXArray", "dataclass", "field"]

import dataclasses
from typing import Any, Callable, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np

JAXArray = Union[np.ndarray, jnp.ndarray]

# This section is based closely on the implementation in flax:
#
# https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/flax/struct.py
#
# This decorator is interpreted by static analysis tools as a hint
# that a decorator or metaclass causes dataclass-like behavior.
# See https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md
# for more information about the __dataclass_transform__ magic.
_T = TypeVar("_T")


def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
    # If used within a stub file, the following implementation can be
    # replaced with "...".
    return lambda a: a


@__dataclass_transform__()
def dataclass(clz: Type[Any]) -> Type[Any]:
    data_clz: Any = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)

    def replace(self: Any, **updates: _T) -> _T:
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    def iterate_clz(x: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta: Tuple[Any, ...], data: Tuple[Any, ...]) -> Any:
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(
        data_clz, iterate_clz, clz_from_iterable
    )

    # Hack to make this class act as a tuple when unpacked
    data_clz.iter_elems = lambda self: iterate_clz(self)[0].__iter__()

    return data_clz


def field(pytree_node: bool = True, **kwargs: Any) -> Any:
    return dataclasses.field(metadata={"pytree_node": pytree_node}, **kwargs)
