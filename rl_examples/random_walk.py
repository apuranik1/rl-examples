from enum import Enum, auto
from random import randrange
from typing import Sequence

import numpy as np

from rl_examples.approximation import Featurizer
from rl_examples.psfa import PSFAEnvironment, State


class LineWalkState(State):
    def __init__(self, position: int, terminal: bool):
        super().__init__(terminal)
        self.position = position

    def __repr__(self) -> str:
        return "LineWalkState({}, {})".format(self.position, self.terminal)


class LineWalkAction(Enum):
    """One one action, which is to wait and let the environment move"""

    NOOP = auto()


class LineWalkEnvironment(PSFAEnvironment[LineWalkState, LineWalkAction]):
    """An environment representing a random walk on a line.
    There is only one action, NOOP, which lets the environment randomly move.
    """

    def __init__(self, nstates: int = 1000, max_jump: int = 100):
        if nstates < 1:
            raise ValueError("Must have at least 1 state")
        self.n = nstates
        self.position = (nstates + 1) // 2
        self.max_jump = max_jump

    @property
    def state(self) -> LineWalkState:
        terminal = self.position == 0 or self.position == self.n + 1
        return LineWalkState(self.position, terminal)

    def reset(self) -> None:
        self.position = (self.n + 1) // 2

    def get_actions(self, state: LineWalkState = None) -> Sequence[LineWalkAction]:
        return [LineWalkAction.NOOP]

    def take_action(self, action: LineWalkAction) -> float:
        distance = randrange(self.max_jump)
        dest = self.position - distance if randrange(2) else self.position + distance
        if dest <= 0:
            self.position = 0
            return -1.0
        elif dest > self.n:
            self.position = self.n + 1
            return 1.0
        else:
            self.position = dest
            return 0.0


class LineWalkFeaturizer(Featurizer[LineWalkState]):
    def featurize(self, data: LineWalkState) -> np.ndarray:
        return np.array([data.position])

    def output_shape(self) -> Sequence[int]:
        return [1]
