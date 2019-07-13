from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Sequence

from rl_examples.discrete import DiscreteEnvironment, State


@dataclass(frozen=True)
class CliffState(State):
    x: int
    y: int


class Move(Enum):
    Up = auto()
    Down = auto()
    Left = auto()
    Right = auto()


class CliffWalkEnvironment(DiscreteEnvironment[CliffState, Move]):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # only terminal state is bottom right corner
        self.state_grid = [
            [CliffState((x, y) == (width - 1, 0), x, y) for y in range(height)]
            for x in range(width)
        ]
        self._state = self.state_grid[0][0]

    @property
    def state(self) -> CliffState:
        return self._state

    def reset(self) -> None:
        self._state = self.state_grid[0][0]

    def state_list(self) -> Sequence[CliffState]:
        return sum(self.state_grid, [])

    def get_actions(self, state: CliffState = None) -> List[Move]:
        actions: List[Move] = []
        _state = state or self._state
        if _state.x > 0:
            actions.append(Move.Left)
        if _state.y >= 0:
            actions.append(Move.Down)  # can go off edge
        if _state.x < self.width - 1:
            actions.append(Move.Right)
        if _state.y < self.height - 1:
            actions.append(Move.Up)
        return actions

    def take_action(self, action: Move) -> float:
        x, y = self._state.x, self._state.y
        if action == Move.Up:
            new_x, new_y = x, y + 1
        elif action == Move.Down:
            new_x, new_y = x, y - 1
        elif action == Move.Left:
            new_x, new_y = x - 1, y
        else:
            new_x, new_y = x + 1, y
        # constant return of -1 incentivizes finishing as fast as possible
        if new_y == -1:
            self._state = self.state_grid[0][0]
            return -100.0
        else:
            self._state = self.state_grid[new_x][new_y]
            return -1.0
