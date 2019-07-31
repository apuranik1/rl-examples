"""Linear approximation algorithms"""

from collections import deque
from functools import reduce
from operator import mul
from typing import Callable, Deque, List, Sequence, Tuple, TypeVar

import numpy as np

from rl_examples.psfa import PSFAAgent, PSFAEnvironment, State
from rl_examples.approximation import TrainableEstimator, Featurizer

TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


class LinearEstimator(TrainableEstimator):
    def __init__(self, input_shape: Sequence[int], lr: float):
        flat_dim = reduce(mul, input_shape, 1)
        self.weights = np.zeros(flat_dim, np.float32)
        self.bias = 0.0
        self.lr = lr

    def batch_estimate(self, data: np.ndarray) -> np.ndarray:
        batch_size = data.shape[0]
        return data.reshape(batch_size, -1).dot(self.weights) + self.bias

    def batch_estimate_and_update(
        self, data: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        batch_size = data.shape[0]
        features = data.reshape(batch_size, -1)
        estimate = features.dot(self.weights) + self.bias
        error = estimate - target
        weight_grad = error.dot(features) / batch_size
        self.weights -= self.lr * weight_grad
        self.bias -= self.lr * error.sum() / batch_size
        return estimate


class LinearApproximationMCAgent(PSFAAgent[TState, TAction]):
    """An approximation-only agent which does not update its policy."""

    def __init__(
        self,
        env: PSFAEnvironment[TState, TAction],
        policy: Callable[[TState], TAction],
        featurizer: Featurizer[TState],
        reward_decay: float,
        lr: float,
    ):
        self.env = env
        self.policy = policy
        self.featurizer = featurizer
        self.gamma = reward_decay
        feature_shape = self.featurizer.output_shape()
        self.estimator = LinearEstimator(feature_shape, lr)
        self.episode_data: List[Tuple[TState, float]] = []

    def action(self, state: TState) -> TAction:
        return self.policy(state)

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action = self.action(state)
        reward = self.env.take_action(action)
        self.episode_data.append((state, reward))
        return state, action, reward

    def episode_end(self) -> None:
        trailing = 0.0
        for state, reward in reversed(self.episode_data):
            trailing = reward + self.gamma * trailing
            self.estimator.estimate_and_update(state, trailing)
        self.episode_data.clear()


class LinearApproximationTDNAgent(PSFAAgent[TState, TAction]):
    """A bootstrapping approximation-only agent which does not update its policy."""

    def __init__(
        self,
        env: PSFAEnvironment[TState, TAction],
        policy: Callable[[TState], TAction],
        featurizer: Featurizer[TState],
        reward_decay: float,
        n: int,
        lr: float,
    ):
        self.env = env
        self.policy = policy
        self.featurizer = featurizer
        self.gamma = reward_decay
        self.n = n
        self.estimator = LinearEstimator(self.featurizer.output_shape(), lr)
        self.data_queue: Deque[Tuple[TState, float]] = deque()

    def action(self, state: TState) -> TAction:
        return self.policy(state)

    def _evaluate_queue(self, trailing_rewards: float) -> float:
        value = 0.0
        discount_factor = 1.0
        for state, reward in self.data_queue:
            value += discount_factor * reward
            discount_factor *= self.gamma
        return value + discount_factor * trailing_rewards

    def _featurize(self, state: TState) -> np.ndarray:
        return self.featurizer.featurize(state)

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action = self.action(state)
        reward = self.env.take_action(action)
        if len(self.data_queue) == self.n:
            trailing_estimate = self.estimator.estimate(self._featurize(state))
            reward_estimate = self._evaluate_queue(trailing_estimate)
            old_state, old_reward = self.data_queue.popleft()
            self.estimator.estimate_and_update(
                self._featurize(old_state), reward_estimate
            )
        self.data_queue.append((state, reward))
        return state, action, reward

    def episode_end(self) -> None:
        while self.data_queue:
            reward_estimate = self._evaluate_queue(0.0)
            old_state, old_reward = self.data_queue.popleft()
            self.estimator.estimate_and_update(old_state, reward_estimate)


def run_example() -> None:
    import rl_examples.random_walk as rw
    from rl_examples.psfa import train

    env = rw.LineWalkEnvironment(1000, 100)
    agent = LinearApproximationTDNAgent(
        env, lambda s: rw.LineWalkAction.NOOP, rw.LineWalkFeaturizer(), 1.0, 5, 2e-8
    )
    train(env, agent, 100, 1000)
    print(agent.estimator.weights)


if __name__ == "__main__":
    run_example()
