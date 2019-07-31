from typing import Deque, Tuple
from collections import deque

import numpy as np

from .psfa import PSFAAgent, PSFAEnvironment, TState, TAction
from .approximation import TrainableEstimator, Featurizer


class ApproximationTDNAgent(PSFAAgent[TState, TAction]):
    """A bootstrapping agent using n-step SARSA"""

    def __init__(
        self,
        env: PSFAEnvironment[TState, TAction],
        featurizer: Featurizer[Tuple[TState, TAction]],
        estimator: TrainableEstimator,
        exploration_rate: float,
        n: int,
        lr: float,
        use_average_reward: bool = False,
    ):
        self.env = env
        self.featurizer = featurizer
        self.estimator = estimator
        self.differential = use_average_reward
        self.avg_reward = 0.0
        self.epsilon = exploration_rate
        self.n = n
        self.lr = lr
        self.data_queue: Deque[Tuple[TState, TAction, float]] = deque()

    def action(self, state: TState) -> TAction:
        available_actions = self.env.get_actions(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # type: ignore
        else:
            batch_featurized = np.stack(
                [self._featurize(state, action) for action in available_actions]
            )
            value_estimates = self.estimator.batch_estimate(batch_featurized)
            max_idx: int = np.argmax(value_estimates)
            return available_actions[max_idx]

    def _evaluate_queue(self, trailing_rewards: float) -> float:
        est = trailing_rewards + sum(reward for s, a, reward in self.data_queue)
        if self.differential:
            return est - len(self.data_queue) * self.avg_reward
        else:
            return est

    def _featurize(self, state: TState, action: TAction) -> np.ndarray:
        return self.featurizer.featurize((state, action))

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action = self.action(state)
        reward = self.env.take_action(action)
        if len(self.data_queue) == self.n:
            trailing_estimate = self.estimator.estimate(self._featurize(state, action))
            reward_estimate = self._evaluate_queue(trailing_estimate)
            old_state, old_action, old_reward = self.data_queue.popleft()
            current_estimate = self.estimator.estimate_and_update(
                self._featurize(old_state, old_action), reward_estimate
            )
            self.avg_reward += self.lr * (reward_estimate - current_estimate)
        self.data_queue.append((state, action, reward))
        return state, action, reward

    def episode_end(self) -> None:
        while self.data_queue:
            reward_estimate = self._evaluate_queue(0.0)
            old_state, old_action, old_reward = self.data_queue.popleft()
            current_estimate = self.estimator.estimate_and_update(
                self._featurize(old_state, old_action), reward_estimate
            )
            self.avg_reward += self.lr * (reward_estimate - current_estimate)
