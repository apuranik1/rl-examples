from typing import Dict, TypeVar

import numpy as np

from rl_examples.discrete import DiscreteEnvironment, DiscreteAgent, State


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


def arbitrary_policy(
    env: DiscreteEnvironment[TState, TAction]
) -> Dict[TState, TAction]:
    return {s: next(iter(env.get_actions(s))) for s in env.nonterminal_states()}


class TabularAgent(DiscreteAgent[TState, TAction]):
    def __init__(self, env: DiscreteEnvironment[TState, TAction], reward_decay: float):
        self.env = env
        self.gamma = reward_decay
        states = self.env.nonterminal_states()
        self.estimates = {
            (state, act): 0.0 for state in states for act in env.get_actions(state)
        }


class GreedyAgent(TabularAgent[TState, TAction]):
    """Implements `action` for a greedy agent"""

    def __init__(self, env: DiscreteEnvironment[TState, TAction], reward_decay: float):
        super().__init__(env, reward_decay)
        self.policy = arbitrary_policy(env)

    def action(self, state: TState) -> TAction:
        return self.policy[state]

    def _optimize_policy(self, state: TState) -> None:
        """Greedily optimize the policy's behavior at `state`."""
        self.policy[state] = max(
            self.env.get_actions(state), key=lambda a: self.estimates[(state, a)]
        )


class EpsilonGreedyAgent(TabularAgent[TState, TAction]):
    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
    ):
        super().__init__(env, reward_decay)
        self.epsilon = exploration_rate
        self.policy = arbitrary_policy(env)

    def action(self, state: TState) -> TAction:
        actions = self.env.get_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)  # type: ignore
        return self.policy[state]

    def _optimize_policy(self, state: TState) -> None:
        """Greedily optimize the policy's behavior at `state`."""
        self.policy[state] = max(
            self.env.get_actions(state), key=lambda a: self.estimates[(state, a)]
        )
