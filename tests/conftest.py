# -*- coding: utf-8 -*-
# mypy: ignore-errors

from jax.config import config

config.update("jax_enable_x64", True)
