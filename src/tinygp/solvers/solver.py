# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Solver"]

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from tinygp.helpers import JAXArray
from tinygp.kernels import Kernel


class Solver(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def init(
        cls,
        kernel: Kernel,
        X: JAXArray,
        diag: JAXArray,
        *,
        covariance: Optional[Any] = None,
    ) -> "Solver":
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def covariance(self) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def normalization(self) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def solve_triangular(
        self, y: JAXArray, *, transpose: bool = False
    ) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def dot_triangular(self, y: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def condition(
        self,
        kernel: Kernel,
        X_test: Optional[JAXArray],
        diag: Optional[JAXArray],
    ) -> Any:
        raise NotImplementedError
