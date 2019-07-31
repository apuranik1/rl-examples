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
        est: float = self.batch_estimate(data[None])[0]
        return est

    @abc.abstractmethod
    def batch_estimate(self, data: np.ndarray) -> np.ndarray:
        """Compute estimates given the input data

        data: ndarray of shape [batch_size, ...]
        returns: float ndarray of shape [batch_size]
        """
        return np.array([self.estimate(data[i]) for i in range(data.shape[0])])


class TrainableEstimator(Estimator):
    def estimate_and_update(self, data: np.ndarray, target: float) -> float:
        """Computes an estimate and updates parameters based on the input and target"""
        est: float = self.batch_estimate_and_update(data[None], np.array([target]))[0]
        return est

    @abc.abstractmethod
    def batch_estimate_and_update(
        self, data: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Compute estimates given the input data, and update based on the targets.
        Returns the original estimates.

        data: ndarray of shape [batch_size, ...]
        targets: ndarray of shape [batch_size]
        returns: float ndarray of shape [batch_size]
        """
        return np.array(
            [
                self.estimate_and_update(data[i], targets[i])
                for i in range(data.shape[0])
            ]
        )
