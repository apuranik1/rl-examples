import abc
from typing import Generic, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


class Featurizer(Generic[T], abc.ABC):
    @abc.abstractmethod
    def featurize(self, data: T) -> np.ndarray:
        pass

    @abc.abstractmethod
    def output_shape(self) -> Sequence[int]:
        pass


class Estimator(abc.ABC):
    def estimate(self, data: np.ndarray) -> float:
        """Compute an estimate given the input data"""
        pass

    def estimate_and_update(self, data: np.ndarray, target: float) -> float:
        """Compute an estimate given the input data, and update based on the target.
        Returns the original estimate.
        """
        pass
