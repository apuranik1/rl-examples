"""More or less a lead-in to Q-learning.

An interesting observation - due to the winner's curse, if our rewards are noisy
we'll frequently overestimate state values.
"""

import abc
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np


@dataclass(frozen=True)
class State:
    terminal: bool


TAction = TypeVar("TAction")
TState = TypeVar("TState", bound=State)


class Environment(abc.ABC, Generic[TAction, TState]):
    """An abstract base class defining the interface for a finite environment.
    Useful for problems with a tractable number of distinct states that can be
    easily simulated, but not easily described by an MDP.
    """

    @property
    @abc.abstractmethod
    def state(self) -> TState:
        """Return the current state"""
        pass

    @property
    def terminated(self) -> bool:
        return self.state.terminal

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def state_list(self) -> Sequence[TState]:
        """Return a list of all possible states"""
        pass

    def nonterminal_states(self) -> Sequence[TState]:
        return [s for s in self.state_list() if not s.terminal]

    def terminal_states(self) -> Sequence[TState]:
        return [s for s in self.state_list() if not s.terminal]

    @abc.abstractmethod
    def get_actions(self, state: TState = None) -> Iterable[TAction]:
        """Return an iterable of available actions at state.

        If state is not specified, default to the current state.
        """
        pass

    def take_action(self, action: TAction) -> float:
        """Respond to an actor taking an action, returning the reward"""
        pass


def _arbitrary_policy(env: Environment[TAction, TState]) -> Dict[TState, TAction]:
    return {s: next(iter(env.get_actions(s))) for s in env.nonterminal_states()}


class Agent(Generic[TAction, TState]):
    """An base class for an agent learning in an environment"""

    def __init__(self, env: Environment[TAction, TState]):
        self.env = env

    def action(self) -> TAction:
        raise NotImplementedError()

    def update(self, oldstate: TState, action: TAction, reward: float, t: int) -> None:
        raise NotImplementedError()

    def episode_end(self) -> None:
        pass


class MonteCarloGreedy(Agent[TAction, TState]):
    """A greedy on-policy Monte Carlo control agent which does no exploration.

    Only tracks return averages after the first state visit in an epoch.

    The lack of exploration makes this a really bad algorithm, so it does not
    provide an action() implementation. Subclasses should override action.
    """

    def __init__(self, env: Environment[TAction, TState], reward_decay: float):
        super().__init__(env)
        self.gamma = reward_decay
        states = env.state_list()
        self.observation_counts = {
            (state, act): 0 for state in states for act in env.get_actions(state)
        }
        self.estimates = {index: 0.0 for index in self.observation_counts}
        self.policy = _arbitrary_policy(self.env)
        # map (s, a) to first index in episode
        self.first_in_episode: Dict[Tuple[TState, TAction], int] = {}
        # track (s, a, reward) for each episode
        self.episode_data: List[Tuple[TState, TAction, float]] = []

    def greedy_action(self, state: TState) -> TAction:
        return self.policy[state]

    def update(self, oldstate: TState, action: TAction, reward: float, t: int) -> None:
        # record the data, but don't act on it
        index = (oldstate, action)
        if index not in self.first_in_episode:
            self.first_in_episode[index] = len(self.episode_data)
        self.episode_data.append((oldstate, action, reward))

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


class MonteCarloES(MonteCarloGreedy[TAction, TState]):
    """Monte Carlo greedy agent with exploring starts"""

    def action(self) -> TAction:
        if len(self.episode_data) == 0:
            # pick an action at random (exploring start)
            actions = self.env.get_actions()
            return np.random.choice(list(actions))  # type: ignore
        # pick action according to policy
        return self.greedy_action(self.env.state)


class MonteCarloSoft(MonteCarloGreedy[TAction, TState]):
    def __init__(
        self,
        env: Environment[TAction, TState],
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


class MonteCarloOffPolicy(Agent[TAction, TState]):
    """A Monte Carlo control agent which uses importance sampling to translate
    feedback from a behavior policy into estimates for a target policy.
    """

    def __init__(
        self,
        env: Environment[TAction, TState],
        exploration_rate: float,
        reward_decay: float,
    ):
        super().__init__(env)
        self.exploration_rate = exploration_rate
        self.reward_decay = reward_decay
        # set arbitrary policy
        self.policy = _arbitrary_policy(self.env)
        self.estimates = {
            (s, a): 0 for s in env.nonterminal_states() for a in env.get_actions(s)
        }


def train(agent: Agent[TAction, TState], n_episodes: int) -> None:
    env = agent.env
    for ep in range(n_episodes):
        # run until termination
        while not env.terminated:
            selected = agent.action()
            reward = env.take_action(selected)
