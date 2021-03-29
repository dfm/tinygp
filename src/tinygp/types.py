# -*- coding: utf-8 -*-

__all__ = ["JAXArray"]

from typing import Union

import jax.numpy as jnp
import numpy as np

JAXArray = Union[np.ndarray, jnp.ndarray]
