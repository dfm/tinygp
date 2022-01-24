# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Solver"]

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy import linalg

from .kernels import Kernel
from .types import JAXArray


class Solver:
    def __init__(self, kernel: Kernel, X: JAXArray, diag: JAXArray):
        pass

    def solve(self, y: JAXArray) -> Tuple[JAXArray, JAXArray]:
        pass
