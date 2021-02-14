# -*- coding: utf-8 -*-

__all__ = ["compose"]

from functools import reduce


def compose(*functions):
    return reduce(lambda f, g: lambda *args: f(g(*args)), functions)
