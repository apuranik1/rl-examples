import abc
from dataclasses import dataclass
from typing import Generic, Iterable, Sequence, Tuple, TypeVar


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

    def __init__(self, env: DiscreteEnvironment[TState, TAction]):
        self.env = env

    @abc.abstractmethod
    def action(self) -> TAction:
        pass

    @abc.abstractmethod
    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        """Run a single act-and-train step.
        Returns the tuple (S_t, A_t, r_t).
        """
        pass

    def episode_end(self) -> None:
        pass
