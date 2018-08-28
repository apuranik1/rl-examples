"""Implementation of an exact MDP solver.

The full specification of the process includes the states, the actions
possible at each state, the transition probabilities induced by each
action, and the conditional expectation of rewards given the current state,
action, and new state.

An MDP is encoded as a list of states. Each state comprises a list of actions.
Each action is a list of Outcome instances. A terminal state has only one
possible action, which is a self-loop with probability 1 and reward 0.
"""

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


Outcome = namedtuple('Outcome', 'prob state reward')


def value_action(outcomes, values, gamma):
    return sum(o.prob * (o.reward + gamma * values[o.state])
               for o in outcomes)


def evaluate_policy(policy, mdp, init_values, gamma, convergence_eps=0.001):
    values = np.zeros(len(mdp))
    stop = False
    while not stop:
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

        if delta < convergence_eps:
            stop = True
    return values


def update_policy(policy, mdp, values, gamma):
    """Update a policy greedily based on the MDP and values given.

    Returns True if the policy has converged, False if changes were made.
    Note: changes policy in-place.
    """
    # compute the greedy policy based on the mdp
    converged = True
    for state, actions in enumerate(mdp):

        def value_func(a):
            return value_action(actions[a], values, gamma)

        # compute argmax_{a \in actions} value(s, a)
        old_policy = policy[state]
        new_policy = max(range(len(actions)), key=value_func)
        if old_policy != new_policy:
            converged = False
        policy[state] = new_policy
    return converged


def compute_policy(mdp, gamma):
    nstates = len(mdp)
    policy = np.zeros(nstates, dtype=np.int)
    values = np.zeros(nstates)
    converged = False
    while not converged:
        values = evaluate_policy(policy, mdp, values, gamma)
        sns.heatmap(np.reshape(values, (4, 4)), annot=True)
        plt.show()
        converged = update_policy(policy, mdp, values, gamma)
        print(np.reshape(policy, (4, 4)))
        print(np.reshape(values, (4, 4)))
    return policy


def mdp_example():
    gamma = 0.9
    # simple gridworld policy - cells (0,0) and (3,3) are goal states

    def make_state(cell_num):
        if cell_num == 0 or cell_num == 15:
            return [[Outcome(1, cell_num, 0)]]  # terminal state
        x = cell_num % 4
        y = cell_num // 4
        left = (x - 1, y, [Outcome(1, cell_num - 1, -1)])
        right = (x + 1, y, [Outcome(1, cell_num + 1, -1)])
        up = (x, y - 1, [Outcome(1, cell_num - 4, -1)])
        down = (x, y + 1, [Outcome(1, cell_num + 4, -1)])
        return [move[2] for move in [left, right, up, down]
                if (move[0] >= 0 and move[0] < 4
                    and move[1] >= 0 and move[1] < 4)]

    mdp = [make_state(num) for num in range(16)]
    opt = compute_policy(mdp, gamma)
    print(np.reshape(opt, (4, 4)))


if __name__ == '__main__':
    mdp_example()
