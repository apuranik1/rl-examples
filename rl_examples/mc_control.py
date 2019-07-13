"""Monte Carlo value estimation and control

An interesting observation - due to the winner's curse, if our rewards are noisy
we'll frequently overestimate state values.
"""

import abc
from typing import Dict, List, Tuple, TypeVar

import numpy as np

from rl_examples.discrete import (
    arbitrary_policy,
    DiscreteEnvironment,
    DiscreteAgent,
    State,
)


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


class OnPolicyAgent(DiscreteAgent[TState, TAction], abc.ABC):
    @abc.abstractmethod
    def update(self, old_state: TState, action: TAction, reward: float, t: int) -> None:
        pass

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        old_state = self.env.state
        action = self.action()
        reward = self.env.take_action(action)
        self.update(old_state, action, reward, t)
        return (old_state, action, reward)


class OffPolicyAgent(DiscreteAgent[TState, TAction], abc.ABC):
    @abc.abstractmethod
    def action_and_prob_ratio(self) -> Tuple[TAction, float]:
        """A training action, together with the importance sampling ratio.
        The ratio is p_policy(action) / p_behavior(action).
        """
        pass

    @abc.abstractmethod
    def update(
        self,
        old_state: TState,
        action: TAction,
        prob_ratio: float,
        reward: float,
        t: int,
    ) -> None:
        pass

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        old_state = self.env.state
        action, prob_ratio = self.action_and_prob_ratio()
        reward = self.env.take_action(action)
        self.update(old_state, action, prob_ratio, reward, t)
        return (old_state, action, reward)


class MonteCarloGreedy(OnPolicyAgent[TState, TAction]):
    """A greedy on-policy Monte Carlo control agent which does no exploration.

    Only tracks return averages after the first state visit in an epoch.

    The lack of exploration makes this a really bad algorithm, so it does not
    provide an action() implementation. Subclasses should override action.
    """

    def __init__(self, env: DiscreteEnvironment[TState, TAction], reward_decay: float):
        super().__init__(env)
        self.gamma = reward_decay
        states = env.nonterminal_states()
        self.observation_counts = {
            (state, act): 0 for state in states for act in env.get_actions(state)
        }
        self.estimates = {index: 0.0 for index in self.observation_counts}
        self.policy = arbitrary_policy(self.env)
        # map (s, a) to first index in episode
        self.first_in_episode: Dict[Tuple[TState, TAction], int] = {}
        # track (s, a, reward) for each episode
        self.episode_data: List[Tuple[TState, TAction, float]] = []

    def greedy_action(self, state: TState) -> TAction:
        return self.policy[state]

    def update(self, old_state: TState, action: TAction, reward: float, t: int) -> None:
        # record the data, but don't act on it
        index = (old_state, action)
        if index not in self.first_in_episode:
            self.first_in_episode[index] = len(self.episode_data)
        self.episode_data.append((old_state, action, reward))

    def _optimize_policy(self, state: TState) -> None:
        """Greedily optimize the policy's behavior at `state`."""
        self.policy[state] = max(
            self.env.get_actions(state), key=lambda a: self.estimates[(state, a)]
        )

    def episode_end(self) -> None:
        # recompute values and reset saved state from episode
        trailing_rewards = 0.0
        T = len(self.episode_data)
        for i, data in zip(range(T - 1, -1, -1), self.episode_data[::-1]):
            state, action, reward = data
            trailing_rewards = trailing_rewards * self.gamma + reward
            index = (state, action)
            if i == self.first_in_episode[index]:
                # update estimate of rewards from (s,a)
                self.observation_counts[index] += 1
                weight = 1 / self.observation_counts[index]
                self.estimates[index] = (
                    self.estimates[index] * (1 - weight) + trailing_rewards * weight
                )
                self._optimize_policy(state)

        self.episode_data = []
        self.first_in_episode = {}


class MonteCarloES(MonteCarloGreedy[TState, TAction]):
    """Monte Carlo greedy agent with exploring starts"""

    def action(self) -> TAction:
        if len(self.episode_data) == 0:
            # pick an action at random (exploring start)
            actions = self.env.get_actions()
            return np.random.choice(list(actions))  # type: ignore
        # pick action according to policy
        return self.greedy_action(self.env.state)


class MonteCarloSoft(MonteCarloGreedy[TState, TAction]):
    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
    ):
        super().__init__(env, reward_decay)
        self.epsilon = exploration_rate

    def action(self) -> TAction:
        state = self.env.state
        actions = self.env.get_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)  # type: ignore
        return self.greedy_action(state)


class MonteCarloOffPolicy(OffPolicyAgent[TState, TAction]):
    """A Monte Carlo control agent which uses importance sampling to translate
    feedback from a behavior policy into estimates for a target policy.

    The target policy in this case is purely greedy, and the behavior is
    epsilon-greedy.
    This is a rather odd algorithm as it stands, because it can only ever learn
    from the tails of episodes. This happens because off-policy actions have 0
    probability, so they immediately nullify any learning.

    TODO: Discounting-aware importance sampling
    """

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        exploration_rate: float,
        reward_decay: float,
        unbiased: bool = False,
    ):
        super().__init__(env)
        self.exploration_rate = exploration_rate
        self.gamma = reward_decay
        # set arbitrary policy
        self.policy = arbitrary_policy(self.env)
        self.observation_weights = {
            (state, act): 0.0
            for state in env.nonterminal_states()
            for act in env.get_actions(state)
        }  # instead of counts, we now weight by probability
        self.estimates = {(s, a): -1e6 for (s, a) in self.observation_weights}
        # episode_data contains (state, action, prob_ratio, reward) tuples
        self.episode_data: List[Tuple[TState, TAction, float, float]] = []

    def update(
        self,
        old_state: TState,
        action: TAction,
        prob_ratio: float,
        reward: float,
        t: int,
    ) -> None:
        # in theory I should try first-visit vs every-visit
        # but that's more effort
        self.episode_data.append((old_state, action, prob_ratio, reward))

    def _optimize_policy(self, state: TState) -> None:
        """Greedily optimize the policy's behavior at `state`."""
        self.policy[state] = max(
            self.env.get_actions(state), key=lambda a: self.estimates[(state, a)]
        )

    def episode_end(self) -> None:
        trailing_rewards = 0.0
        weight_is = 1.0  # importance sampling ratio
        T = len(self.episode_data)
        for i, data in zip(range(T - 1, -1, -1), self.episode_data[::-1]):
            state, action, prob_ratio, reward = data
            trailing_rewards = trailing_rewards * self.gamma + reward
            index = (state, action)
            # update estimate
            new_weight = self.observation_weights[index] + weight_is
            if new_weight > 0.0:
                # we have actual data to update on
                current_estimate = self.estimates[index]
                new_estimate = current_estimate + (weight_is / new_weight) * (
                    trailing_rewards - current_estimate
                )
                self.estimates[index] = new_estimate
                self.observation_weights[index] = new_weight
                self._optimize_policy(state)
            # weight prior data by the probability of the current move
            weight_is = weight_is * prob_ratio
            if weight_is == 0.0:
                break
        # reset
        self.episode_data = []

    def action_and_prob_ratio(self) -> Tuple[TAction, float]:
        state = self.env.state
        actions = self.env.get_actions()
        policy_action = self.policy[state]
        if np.random.rand() < self.exploration_rate:
            choice = np.random.choice(actions)
            return choice, float(choice == policy_action)
        else:
            return policy_action, 1.0

    def action(self) -> TAction:
        return self.policy[self.env.state]


def run_example() -> None:
    from rl_examples.cliff_walk import CliffWalkEnvironment
    from rl_examples.discrete import train

    env = CliffWalkEnvironment(5, 3)
    agent = MonteCarloOffPolicy(env, 0.2, 0.9)
    print("initial policy")
    print(agent.policy)
    print()
    train(agent, 1000)
    print("ending status")
    print(agent.policy)
    print()
    print(agent.observation_weights)
    print()
    print(agent.estimates)


if __name__ == "__main__":
    run_example()
