"""N-step bootstrapping methods. These are direct generalizations of TD(0) methods.

The N-step delta can be written as a sum of TD(0) deltas, provided values don't update

d_t = R_{t+1} + gamma V(S_{t+1}) - V(S_{t})
d_{n_step} = -V(S_{t}) + R_{t+1} + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n})
= d_t - gamma V(S_{t+1}) + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n})
= d_t + gamma d_{t+1} - gamma^2 V(S_{t+2}) + gamma^2 R_{t+3} + ... + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n})
= d_t + gamma d_{t+1} + gamma^2 d_{t+2} + ... + gamma^{n-1} d_{t+n-1}
"""

from collections import deque
from typing import Deque, Optional, Sequence, Tuple, TypeVar

import numpy as np

from rl_examples.discrete import DiscreteEnvironment, State
from rl_examples.tabular import EpsilonGreedyAgent, GreedyAgent
from rl_examples.td import td_update


TState = TypeVar("TState", bound=State)
TAction = TypeVar("TAction")


class SarsaNAgent(EpsilonGreedyAgent[TState, TAction]):
    """An epsilon-greedy online on-policy agent with n-bootstrap updates"""

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        n: int,
        lr: float = 0.1,
    ):
        super().__init__(env, reward_decay, exploration_rate)
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.lr = lr
        # previous step's action and reward, if applicable
        self.data_queue: Deque[Tuple[TState, TAction, float]] = deque(maxlen=self.n)

    def update(
        self,
        state: TState,
        action: TAction,
        reward: float,
        trailing_reward_estimate: float,
    ) -> None:
        """The SARSA update for an ongoing episode.
        trailing_reward_estimate is generally estimated from the next n timesteps
        """
        td_update(self, state, action, reward, trailing_reward_estimate, self.lr)
        self._optimize_policy(state)

    def _evaluate_queue(self, trailing_reward_estimate: float) -> float:
        # return a valuation of data_queue's rewards
        value = 0.0
        discount_factor = 1.0
        for state, action, reward in self.data_queue:
            value += discount_factor * reward
            discount_factor *= self.gamma
        return value + discount_factor * trailing_reward_estimate

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        # if we haven't already planned on an action, decide now
        state = self.env.state
        action = self.action(state)
        reward = self.env.take_action(action)
        if len(self.data_queue) == self.n:
            # update on the first state/action combo
            old_state, old_action, old_reward = self.data_queue.popleft()
            reward_estimate = self._evaluate_queue(self.estimates[state, action])
            self.update(old_state, old_action, old_reward, reward_estimate)
        self.data_queue.append((state, action, reward))
        return (state, action, reward)

    def episode_end(self) -> None:
        # essentially run monte carlo updates
        trailing_rewards = 0.0
        for state, action, reward in reversed(self.data_queue):
            self.update(state, action, reward, trailing_rewards)
            trailing_rewards = trailing_rewards * self.gamma + reward
        self.data_queue.clear()


# There's also an ExpectedSarsaNAgent, but I don't think it contains new insights
# beyond combining the SarsaNAgent code with the ExpectedSarsaAgent code


class OffPolicySarsaNAgent(GreedyAgent[TState, TAction]):
    """An off-policy agent using n-step bootstrapping.
    Instead of the naive generalization of SARSA-n to off-policy, uses a per-decision
    importance sampling average to reduce variance.
    """

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        n: int,
        lr: float = 0.1,
    ):
        super().__init__(env, reward_decay)
        if n <= 0:
            raise ValueError("n must be a positive integer")
        self.epsilon = exploration_rate
        self.n = n
        self.lr = lr
        self.data_queue: Deque[Tuple[TState, TAction, float, float]] = deque()

    def _evaluate_queue(self, trailing_reward_estimate: float) -> float:
        """Evaluate the queue and return an estimate of its value"""
        value = 0.0
        discount_factor = 1.0
        prob_ratio = 1.0
        for state, action, reward, rho in self.data_queue:
            # add the contribution for this time step
            # the control variate has 0 expected value and follows the greedy policy
            control_variate = (1 - rho) * self.estimates[state, self.policy[state]]
            reward_contrib = rho * reward
            value += prob_ratio * discount_factor * (reward_contrib + control_variate)
            discount_factor *= self.gamma
            prob_ratio *= rho
            if prob_ratio == 0.0:
                break
        value += prob_ratio * discount_factor * trailing_reward_estimate
        return value

    def train_action_with_prob_ratio(self, state: TState) -> Tuple[TAction, float]:
        env_actions = list(self.env.get_actions(state))
        n_actions = len(env_actions)
        prob_on_policy = 1 - (self.epsilon * (1 - 1 / n_actions))
        policy_action = self.action(self.env.state)
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(env_actions)
            return choice, float(choice == policy_action) / prob_on_policy
        else:
            return policy_action, 1 / prob_on_policy

    def update(
        self,
        state: TState,
        action: TAction,
        reward: float,
        trailing_reward_estimate: float,
    ) -> None:
        """The SARSA update for an ongoing episode.
        trailing_reward_estimate is generally estimated from the next n timesteps
        """
        td_update(self, state, action, reward, trailing_reward_estimate, self.lr)
        self._optimize_policy(state)

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action, rho = self.train_action_with_prob_ratio(state)
        reward = self.env.take_action(action)
        # print(state, action, reward)
        if len(self.data_queue) == self.n:
            # update on the first state/action combo
            old_state, old_action, old_reward, old_rho = self.data_queue.popleft()
            last_greedy = self.estimates[state, self.policy[state]]
            reward_estimate = self._evaluate_queue(last_greedy)
            self.update(old_state, old_action, old_reward, reward_estimate)
        self.data_queue.append((state, action, reward, rho))
        return (state, action, reward)

    def episode_end(self) -> None:
        while self.data_queue:
            old_state, old_action, old_reward, old_rho = self.data_queue.popleft()
            reward_estimate = self._evaluate_queue(0.0)
            self.update(old_state, old_action, old_reward, reward_estimate)
        self.data_queue.clear()


OptionsList = Sequence[Tuple[TAction, float]]


class EpsGreedyTreeBackupAgent(EpsilonGreedyAgent[TState, TAction]):
    """An epsilon-greedy agent implementing tree backup.
    It's telling that the code for this algorithm has precisely the same structure
    as that of the SARSA off-policy agent, but with expectations substituted for
    importance-sampled outcomes.

    With a little bit more effort these two classes could certainly share code.
    """

    def __init__(
        self,
        env: DiscreteEnvironment[TState, TAction],
        reward_decay: float,
        exploration_rate: float,
        n: int,
        lr: float,
    ):
        super().__init__(env, reward_decay, exploration_rate)
        self.n = n
        self.lr = lr
        # data_queue holds state, selected action, available actions, and the reward
        self.data_queue: Deque[Tuple[TState, TAction, float]] = deque()

    def options(self, state: TState) -> OptionsList:
        possible_actions = list(self.env.get_actions(state))
        greedy_action = self.policy[state]
        nongreedy_prob = self.epsilon / len(possible_actions)
        greedy_prob = 1 - self.epsilon * (1 - 1 / len(possible_actions))
        options = [
            (a, greedy_prob if a == greedy_action else nongreedy_prob)
            for a in possible_actions
        ]
        return options

    def update(
        self,
        state: TState,
        action: TAction,
        reward: float,
        trailing_reward_estimate: float,
    ) -> None:
        """The SARSA update for an ongoing episode.
        trailing_reward_estimate is generally estimated from the next n timesteps
        """
        td_update(self, state, action, reward, trailing_reward_estimate, self.lr)
        self._optimize_policy(state)

    def _evaluate_queue(self, trailing_reward_estimate: float) -> float:
        value = 0.0
        discount_factor = 1.0
        probability = 1.0
        for state, selected, reward in self.data_queue:
            reward_coef = discount_factor * probability
            current_prob: Optional[float] = None
            for action, prob in self.options(state):
                if action == selected:
                    current_prob = prob
                else:
                    value += reward_coef * prob * self.estimates[state, action]
            if current_prob is None:
                raise ValueError("Selected option not found")
            probability *= current_prob
            discount_factor *= self.gamma
        return value + discount_factor * probability * trailing_reward_estimate

    def expected_value(self, state: TState) -> float:
        return sum(prob * self.estimates[state, a] for a, prob in self.options(state))

    def act_and_train(self, t: int) -> Tuple[TState, TAction, float]:
        state = self.env.state
        action = self.action(state)
        reward = self.env.take_action(action)
        if len(self.data_queue) == self.n:
            old_state, old_action, old_reward = self.data_queue.popleft()
            expected_last = self.expected_value(state)
            reward_estimate = self._evaluate_queue(expected_last)
            self.update(old_state, old_action, old_reward, reward_estimate)
        self.data_queue.append((state, action, reward))
        return (state, action, reward)

    def episode_end(self) -> None:
        while self.data_queue:
            old_state, old_action, old_reward = self.data_queue.popleft()
            reward_estimate = self._evaluate_queue(0.0)
            self.update(old_state, old_action, old_reward, reward_estimate)
        self.data_queue.clear()


# not implemented: Q(sigma) learning which smoothly transitions between off-policy
# SARSA and tree backup depending on the value of sigma.


def run_example() -> None:
    from pprint import pprint
    from rl_examples.cliff_walk import CliffWalkEnvironment
    from rl_examples.discrete import train

    env = CliffWalkEnvironment(8, 5)
    # larger values of n seem to require a higher learning rate
    # 1.0 actually works well here (since environment is deterministic)
    agent = OffPolicySarsaNAgent(env, 1.0, 0.1, 2, 1.0)
    i = 0

    def on_episode_end(t: int) -> None:
        nonlocal i
        i += 1
        if i % 100 == 0:
            print(t)

    print("training: 1000 episodes")
    train(env, agent, 1000, on_episode_end=on_episode_end)
    print("ending estimates")
    pprint(agent.policy)
    pprint(agent.estimates)

    print("Running in eval mode")
    t = 0
    reward = 0.0
    while not env.terminated:
        reward += env.take_action(agent.action(env.state))
        t += 1
    env.reset()
    print(t, reward)


if __name__ == "__main__":
    run_example()
