import abc
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Sequence, Tuple, TypeVar


@dataclass(frozen=True)
class State:
    terminal: bool


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


class DiscreteEnvironment(abc.ABC, Generic[TState, TAction]):
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


class DiscreteAgent(Generic[TState, TAction], abc.ABC):
    """An base class for an agent learning in an environment"""

    @abc.abstractmethod
    def action(self, state: TState) -> TAction:
        pass

    @abc.abstractmethod
    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        """Run a single act-and-train step.
        Returns the tuple (S_t, A_t, R_{t+1}).
        """
        pass

    def episode_end(self) -> None:
        pass


def train(
    env: DiscreteEnvironment[TState, TAction],
    agent: DiscreteAgent[TState, TAction],
    n_episodes: int,
    on_action: Callable[[TState, TAction, float, int], None] = None,
    on_episode_end: Callable[[int], None] = None,
) -> None:
    """Trains the agent in its environment.
    At the end of each timestep, calls on_action(S_t, A_t, R_{t+1}, t)
    At the end of each episode, calls on_episode_end(T)
    """
    for ep in range(n_episodes):
        t = 0
        while not env.terminated:
            s, a, r = agent.act_and_train(t)  # returns (S_t, A_t, R_t)
            if on_action:
                on_action(s, a, r, t)
            t += 1
        agent.episode_end()
        if on_episode_end:
            on_episode_end(t)
        env.reset()
