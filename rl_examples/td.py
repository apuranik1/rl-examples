"""Temporal Difference methods, including SARSA and Q-learning"""

import abc
from typing import Optional, Tuple, TypeVar

import numpy as np

from rl_examples.discrete import DiscreteEnvironment, State
from rl_examples.tabular import (
    arbitrary_policy,
    GreedyAgent,
    EpsilonGreedyAgent,
    TabularAgent,
)


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


def td_update(
    agent: TabularAgent[TState, TAction],
    state: TState,
    action: TAction,
    reward: float,
    trailing_reward_estimate: float,
    lr: float,
) -> None:
    """Update Q estimates using TD(0)

    trailing_reward_estimate should be Q(S_{t+1}, A_{t+1}) if S_{t+1} is nonterminal
    In a terminal state it should be 0.
    """
    index = (state, action)
    old_estimate = agent.estimates[index]
    new_estimate = reward + agent.gamma * trailing_reward_estimate
    agent.estimates[index] = old_estimate + lr * (new_estimate - old_estimate)


class SarsaAgent(EpsilonGreedyAgent[TState, TAction]):
    """An epsilon-greedy online on-policy agent (SARSA updates)"""

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        lr: float = 0.1,
    ):
        super().__init__(env, reward_decay, exploration_rate)
        self.lr = lr
        # previous step's action and reward, if applicable
        self.next_action: Optional[TAction] = None

    def update(
        self,
        state: TState,
        action: TAction,
        reward: float,
        trailing_reward_estimate: float,
    ) -> None:
        """The SARSA update for an ongoing episode.

        trailing_reward_estimate should be Q(S_{t+1}, A_{t+1}) if S_{t+1} is nonterminal
        In a terminal state it should be 0.
        """
        td_update(self, state, action, reward, trailing_reward_estimate, self.lr)
        self._optimize_policy(state)

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        # if we haven't already planned on an action, decide now
        old_state = self.env.state
        action = self.next_action or self.action(old_state)
        reward = self.env.take_action(action)
        new_state = self.env.state
        if new_state.terminal:
            self.next_action = None
            self.update(old_state, action, reward, 0.0)
        else:
            self.next_action = self.action(new_state)
            estimate = self.estimates[new_state, self.next_action]
            self.update(old_state, action, reward, estimate)
        return old_state, action, reward

    def episode_end(self) -> None:
        self.next_action = None  # in case we get weirdly interrupted


class QAgent(GreedyAgent[TState, TAction]):
    """A greedy online off-policy agent (Q-learning)"""

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        lr: float = 0.1,
    ):
        super().__init__(env, reward_decay)
        self.epsilon = exploration_rate
        self.lr = lr

    def update(
        self,
        state: TState,
        action: TAction,
        reward: float,
        next_action: Optional[TAction],
    ) -> None:
        new_state = self.env.state
        if new_state.terminal:
            trailing_estimate = 0.0
        else:
            if next_action is None:
                raise ValueError("Must provide next_action in nonterminal state")
            trailing_estimate = self.estimates[new_state, next_action]
        td_update(self, state, action, reward, trailing_estimate, self.lr)
        self._optimize_policy(state)

    def training_action(self, state: TState) -> TAction:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_actions())  # type: ignore
        else:
            return self.policy[state]

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        choice = self.training_action(state)
        reward = self.env.take_action(choice)
        next_action = None if self.env.terminated else self.policy[self.env.state]
        self.update(state, choice, reward, next_action)
        return state, choice, reward


class ExpectedSarsaAgent(TabularAgent[TState, TAction]):
    """An agent implementing Expected SARSA.
    This is a very general algorithm, having Q-learning as a special case.
    As a result methods are broken down finely so subclasses can specialize.
    """

    def __init__(
        self, env: DiscreteEnvironment[TState, TAction], reward_decay: float, lr: float
    ):
        super().__init__(env, reward_decay)
        self.lr = lr

    @abc.abstractmethod
    def expected_reward(self, state: TState) -> float:
        """Compute the expected trailing reward from the given state under the
        target policy.
        """
        pass

    @abc.abstractmethod
    def training_action(self, state: TState, t: int) -> TAction:
        pass

    @abc.abstractmethod
    def _update_policy(self, state: TState, t: int) -> None:
        """Refine the current policy based on estimated values"""
        pass

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        old_state = self.env.state
        action = self.training_action(old_state, t)
        reward = self.env.take_action(action)
        expected_trailing = self.expected_reward(self.env.state)
        td_update(self, old_state, action, reward, expected_trailing, self.lr)
        self._update_policy(old_state, t)
        return old_state, action, reward


class EpsGreedyExpectedSarsaAgent(ExpectedSarsaAgent[TState, TAction]):
    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        lr: float,
        exploration_rate: float,
    ):
        super().__init__(env, reward_decay, lr)
        self.policy = arbitrary_policy(self.env)
        self.epsilon = exploration_rate

    def expected_reward(self, state: TState) -> float:
        if state.terminal:
            return 0.0
        policy_estimate = self.estimates[state, self.policy[state]]
        available_actions = list(self.env.get_actions(state))
        exploration_estimate = sum(
            self.estimates[state, a] for a in available_actions
        ) / len(available_actions)
        return (
            1 - self.epsilon
        ) * policy_estimate + self.epsilon * exploration_estimate

    def training_action(self, state: TState, t: int) -> TAction:
        state = self.env.state
        actions = self.env.get_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)  # type: ignore
        return self.policy[state]

    def action(self, state: TState) -> TAction:
        return self.policy[state]

    def _update_policy(self, state: TState, t: int) -> None:
        self.policy[state] = max(
            self.env.get_actions(state), key=lambda a: self.estimates[state, a]
        )


class DoubleQAgent(GreedyAgent[TState, TAction]):
    """Implements Double Q-Learning.

    The Q-learning update estimates the trailing rewards based on the greedy action.
    However, the greedy action requires maximizing rewards, so it is a biased estimator.
    A double Q agent uses one agent to pick a greedy followup action, and another to
    estimate its value, which mitigates the bias in the estimate.

    Experimentally, this seems to converge much more slowly than Q-learning or SARSA.
    A high exploration rate (around 0.5) seems to mitigate this.
    """

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        lr: float,
    ):
        super().__init__(env, reward_decay)
        self.epsilon = exploration_rate
        self.agents = (
            QAgent(env, reward_decay, exploration_rate, lr),
            QAgent(env, reward_decay, exploration_rate, lr),
        )

    def training_action(self, state: TState) -> TAction:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_actions())  # type: ignore
        else:
            return self.policy[state]

    def update_estimate_and_policy(self, state: TState, action: TAction) -> None:
        index = (state, action)
        self.estimates[index] = 0.5 * (
            self.agents[0].estimates[index] + self.agents[1].estimates[index]
        )
        self._optimize_policy(state)

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action = self.training_action(state)
        reward = self.env.take_action(action)
        # estimator is updated, selector picks the next action
        if np.random.randint(2):
            estimator, selector = self.agents
        else:
            selector, estimator = self.agents
        next_action = None if self.env.terminated else selector.action(self.env.state)
        estimator.update(state, action, reward, next_action)
        self.update_estimate_and_policy(state, action)
        return (state, action, reward)


def run_example() -> None:
    from pprint import pprint
    from rl_examples.cliff_walk import CliffWalkEnvironment
    from rl_examples.discrete import train

    env = CliffWalkEnvironment(8, 3)
    agent = DoubleQAgent(env, 1, lr=0.5, exploration_rate=0.5)
    i = 0

    def on_epoch_end(t: int) -> None:
        nonlocal i
        i += 1
        if i % 100 == 0:
            print(t)

    train(env, agent, 1000, on_episode_end=on_epoch_end)

    # evaluate trained model
    t = 0
    reward = 0.0
    while not env.terminated:
        reward += env.take_action(agent.action(env.state))
        t += 1
    env.reset()
    print(t, reward)
    pprint(agent.estimates)
    # pprint(agent.policy)


if __name__ == "__main__":
    run_example()
