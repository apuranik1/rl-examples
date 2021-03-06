import math
from enum import Enum
from typing import Sequence, Tuple

import numpy as np

from rl_examples.approximation import Featurizer
from rl_examples.psfa import PSFAEnvironment, State


class MountainCarState(State):
    def __init__(self, x: float, vx: float, terminal: bool):
        super().__init__(terminal)
        self.x = x
        self.vx = vx
        self.y = math.sin(3 * self.x)

    def g_x(self) -> float:
        """Compute the acceleration due to gravity"""
        # corresponds to no physical reality
        return -2.5e-3 * math.cos(3 * self.x)

    def __str__(self) -> str:
        return f"MountainCarState(x={self.x}, vx={self.vx})"


class MountainCarAction(Enum):
    Forward = 1
    Neutral = 0
    Reverse = -1


class MountainCarEnvironment(PSFAEnvironment[MountainCarState, MountainCarAction]):

    MIN_X = -1.2
    MAX_X = 0.5
    MIN_VX = -0.07
    MAX_VX = 0.07

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # randomly select starting position in [-0.6, 0.4)
        x = np.random.rand() - 0.6
        self._state = self.make_state(x, 0.0)

    @property
    def state(self) -> MountainCarState:
        return self._state

    def get_actions(
        self, state: MountainCarState = None
    ) -> Sequence[MountainCarAction]:
        return list(MountainCarAction)

    def take_action(self, action: MountainCarAction) -> float:
        old = self._state
        new_vx = np.clip(
            old.vx + 0.001 * action.value + old.g_x(),
            MountainCarEnvironment.MIN_VX,
            MountainCarEnvironment.MAX_VX,
        )
        new_x = old.x + new_vx
        self._state = self.make_state(new_x, new_vx)
        return -1.0

    @staticmethod
    def make_state(x: float, vx: float) -> MountainCarState:
        if x >= MountainCarEnvironment.MAX_X:
            return MountainCarState(MountainCarEnvironment.MAX_X, 0.0, True)
        elif x <= MountainCarEnvironment.MIN_X:
            return MountainCarState(MountainCarEnvironment.MIN_X, 0.0, False)
        else:
            return MountainCarState(x, vx, False)


class MountainCarFeaturizer(Featurizer[Tuple[MountainCarState, MountainCarAction]]):
    def __init__(self) -> None:
        self.indices = {action: i for i, action in enumerate(MountainCarAction)}
        self.action_len = len(self.indices)

    def featurize(self, data: Tuple[MountainCarState, MountainCarAction]) -> np.ndarray:
        """Build the feature array [x, v_x, acceleration]"""
        state, action = data
        action_data = np.zeros(self.action_len)
        action_data[self.indices[action]] = 1
        return np.concatenate([action_data, [state.x, 100 * state.vx]])

    def output_shape(self) -> Sequence[int]:
        return [self.action_len + 2]
