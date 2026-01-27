from contextlib import contextmanager
from typing import Any

import jax
from jax._src.public_test_util import check_close

from tinygp.helpers import JAXArray


def assert_allclose(
    calculated: JAXArray, expected: JAXArray, *args: Any, **kwargs: Any
):
    kwargs["atol"] = kwargs.get(
        "atol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    kwargs["rtol"] = kwargs.get(
        "rtol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    check_close(calculated, expected, *args, **kwargs)


def assert_pytrees_allclose(calculated: Any, expected: Any, *args: Any, **kwargs: Any):
    jax.tree_util.tree_map(
        lambda a, b: assert_allclose(a, b, *args, **kwargs), calculated, expected
    )


def _as_context_manager(obj):
    # If it's already a context manager
    if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
        return obj

    # If it's a generator, wrap it
    if hasattr(obj, "__iter__") and hasattr(obj, "send"):
        return contextmanager(lambda: obj)()

    raise TypeError("Object is neither a context manager nor a generator")


@contextmanager
def jax_enable_x64():
    if hasattr(jax, "enable_x64"):
        cm = jax.enable_x64(True)
    else:
        # deprecated in jax>=0.9
        from jax.experimental import enable_x64 as _enable_x64
        cm = _enable_x64()

    with _as_context_manager(cm):
        yield
