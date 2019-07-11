"""Implementation of an exact MDP solver.

The full specification of the process includes the states, the actions
possible at each state, the transition probabilities induced by each
action, and the conditional expectation of rewards given the current state,
action, and new state.

An MDP is encoded as a list of states. Each state comprises a list of actions.
Each action is a list of Outcome instances. A terminal state has only one
possible action, which is a self-loop with probability 1 and reward 0.

It might be better to have states be dictionaries of actions instead of lists.
This would allow for actions like "forward" to be consistent across different
states, instead of being encoded by multiple potentially distinct integer
indices. This wouldn't create a particularly large code change.

The functions value_update and value_iteration_iter can be composed to build
many MDP solvers. Value iteration exclusively runs the latter. It may be more
efficient to run multiple iterations of value_update before running
value_iteration_iter (Sutton 83). The function update_policy is for updating
only the policy, without refining value estimates online.
"""

from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Outcome:
    prob: float
    state: int
    reward: float


Action = Sequence[Outcome]
State = Sequence[Action]
MDP = Sequence[State]
Policy = List[int]


def value_action(
    outcomes: Sequence[Outcome], values: Sequence[float], gamma: float
) -> float:
    return sum(o.prob * (o.reward + gamma * values[o.state]) for o in outcomes)


def value_update(policy: Policy, mdp: MDP, values: List[float], gamma: float) -> float:
    """Update value estimates of MDP states according to a fixed policy"""
    delta = 0
    for state, actions in enumerate(mdp):
        # evaluate state_i
        selected_action = policy[state]
        outcome_dist = actions[selected_action]
        new_value = value_action(outcome_dist, values, gamma)
        update = np.abs(values[state] - new_value)
        values[state] = new_value
        if update > delta:
            delta = update
    return delta


def value_iteration_iter(mdp: MDP, values: List[float], gamma: float) -> float:
    """Update value estimates of MDP states while taking greedy actions"""
    delta = 0
    for state, actions in enumerate(mdp):
        new_value = max(value_action(a, values, gamma) for a in actions)
        update = np.abs(values[state] - new_value)
        values[state] = new_value
        if update > delta:
            delta = update
    return delta


def evaluate_policy(
    policy: Policy,
    mdp: MDP,
    values: List[float],
    gamma: float,
    convergence_eps: float = 0.001,
) -> List[float]:
    delta = convergence_eps + 1
    while delta >= convergence_eps:
        delta = value_update(policy, mdp, values, gamma)
    return values


def update_policy(
    policy: Policy, mdp: MDP, values: Sequence[float], gamma: float
) -> bool:
    """Update a policy greedily based on the MDP and values given.

    Returns True if the policy has converged, False if changes were made.
    Note: changes policy in-place.
    """
    # compute the greedy policy based on the mdp
    converged = True
    for state, actions in enumerate(mdp):

        def value_func(a: int) -> float:
            return value_action(actions[a], values, gamma)

        # compute argmax_{a \in actions} value(s, a)
        old_policy = policy[state]
        new_policy = max(range(len(actions)), key=value_func)
        if old_policy != new_policy:
            converged = False
        policy[state] = new_policy
    return converged


def compute_policy(mdp: MDP, gamma: float) -> Policy:
    nstates = len(mdp)
    policy: Policy = np.zeros(nstates, dtype=np.int)
    values: List[float] = np.zeros(nstates)
    converged = False
    while not converged:
        values = evaluate_policy(policy, mdp, values, gamma)
        # sns.heatmap(np.reshape(values, (4, 4)), annot=True)
        # plt.show()
        converged = update_policy(policy, mdp, values, gamma)
        # print(np.reshape(policy, (4, 4)))
        # print(np.reshape(values, (4, 4)))
    return policy


def value_iteration(mdp: MDP, gamma: float, epsilon: float = 0.01) -> Policy:
    nstates = len(mdp)
    values = np.zeros(nstates)
    converged = False
    while not converged:
        delta = value_iteration_iter(mdp, values, gamma)
        # plt.plot(values)
        # plt.show()
        converged = delta < epsilon
    policy: List[int] = np.zeros(nstates, dtype=np.int)
    update_policy(policy, mdp, values, gamma)
    return policy


def mdp_example() -> None:
    gamma = 0.9
    # simple gridworld policy - cells (0,0) and (3,3) are goal states

    def make_state(cell_num: int) -> State:
        if cell_num == 0 or cell_num == 15:
            return [[Outcome(1, cell_num, 0)]]  # terminal state
        x = cell_num % 4
        y = cell_num // 4
        left = (x - 1, y, [Outcome(1, cell_num - 1, -1)])
        right = (x + 1, y, [Outcome(1, cell_num + 1, -1)])
        up = (x, y - 1, [Outcome(1, cell_num - 4, -1)])
        down = (x, y + 1, [Outcome(1, cell_num + 4, -1)])
        return [
            move[2]
            for move in [left, right, up, down]
            if (move[0] >= 0 and move[0] < 4 and move[1] >= 0 and move[1] < 4)
        ]

    mdp = [make_state(num) for num in range(16)]
    # opt = compute_policy(mdp, gamma)
    opt = value_iteration(mdp, gamma)
    print(np.reshape(opt, (4, 4)))


def gambler_example() -> None:
    gamma = 0.9

    def make_state(money: int) -> State:
        if money > 100 or money < 0:
            raise ValueError("Money must be in [0, 100]")
        if money == 0:
            return [[Outcome(1, 0, 0)]]
        if money == 100:
            return [[Outcome(1, 100, 0)]]
        actions = []
        for bet in range(0, money + 1):
            positive = min(money + bet, 100)
            pos_reward = 1 if positive == 100 else 0
            negative = money - bet
            neg_reward = 0
            actions.append(
                [Outcome(0.4, positive, pos_reward), Outcome(0.6, negative, neg_reward)]
            )
        return actions

    mdp = [make_state(num) for num in range(101)]
    opt = value_iteration(mdp, gamma)
    plt.plot(opt)
    plt.show()


if __name__ == "__main__":
    # mdp_example()
    gambler_example()
