import abc
from typing import Callable, Generic, Sequence, Tuple, TypeVar


class State:
    def __init__(self, terminal: bool):
        self.terminal = terminal


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


class PSFAEnvironment(abc.ABC, Generic[TState, TAction]):
    """An environment with parametrized states and a finite set of actions (PSFA)
    available at each state.

    Such an environment does not support enumerating all states, but does allow
    enumerating possible actions.
    """

    @property
    @abc.abstractmethod
    def state(self) -> TState:
        pass

    @property
    def terminated(self) -> bool:
        return self.state.terminal

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def get_actions(self, state: TState = None) -> Sequence[TAction]:
        """Return an iterable of available actions at state.

        If state is not specified, default to the current state.
        """
        pass

    @abc.abstractmethod
    def take_action(self, action: TAction) -> float:
        """Respond to an actor taking an action, returning a reward"""
        pass


class PSFAAgent(abc.ABC, Generic[TState, TAction]):
    """A base class for an agent learning in a PSFA environment"""

    @abc.abstractmethod
    def action(self, state: TState) -> TAction:
        pass

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        """Run a single act-and-train step.
        Returns the tuple (S_t, A_t, R_{t+1}).
        """
        pass

    def episode_end(self) -> None:
        pass


def train(
    env: PSFAEnvironment[TState, TAction],
    agent: PSFAAgent[TState, TAction],
    n_episodes: int,
    episode_length_cap: int,
    on_action: Callable[[TState, TAction, float, int], None] = None,
    on_episode_end: Callable[[int], None] = None,
) -> None:
    """Trains the agent in its environment.
    At the end of each timestep, calls on_action(S_t, A_t, R_{t+1}, t)
    At the end of each episode, calls on_episode_end(T)
    """
    for ep in range(n_episodes):
        t = 0
        for step in range(episode_length_cap):
            if env.terminated:
                break
            s, a, r = agent.act_and_train(t)  # returns (S_t, A_t, R_t)
            if on_action:
                on_action(s, a, r, t)
            t += 1
        agent.episode_end()
        if on_episode_end:
            on_episode_end(t)
        env.reset()
