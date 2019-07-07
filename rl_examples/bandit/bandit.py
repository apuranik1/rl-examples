from typing import Sequence

import argh
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from operator import itemgetter


class BanditLearner:
    def __init__(self, actions: Sequence[int], epsilon: float):
        self.eps = epsilon
        self.actions = actions
        self.estimates = {act: 0.0 for act in actions}
        self.action_counts = {act: 0 for act in actions}

    def stepsize(self, action: int, t: int) -> float:
        return 1 / (self.action_counts[action])

    def select_best(self, t: int) -> int:
        return max(self.estimates.items(), key=itemgetter(1))[0]

    def action(self, t: int) -> int:
        rng: float = np.random.rand()
        if rng < self.eps:
            # take a uniformly random action
            return self.actions[np.random.randint(len(self.actions))]  # type: ignore
        return self.select_best(t)

    def update(self, action: int, reward: float, t: int) -> None:
        self.action_counts[action] += 1
        lr = self.stepsize(action, t)
        self.estimates[action] = (1 - lr) * self.estimates[action] + lr * reward


class UCBBandit(BanditLearner):
    def __init__(self, actions: Sequence[int], epsilon: float, confidence: float):
        super().__init__(actions, epsilon)
        self.c = confidence

    def select_best(self, t: int) -> int:
        def upper_bound(act: int) -> float:
            baseline = self.estimates[act]
            uncertainty: float = self.c * np.sqrt(
                np.log(t + 1) / (self.action_counts[act] + 1e-8)
            )
            return baseline + uncertainty

        return max(self.actions, key=upper_bound)


def test_bandits() -> None:
    NUM_ACTIONS = 10
    NUM_TESTS = 1000
    ITERS = 1000
    actions = list(range(10))
    eps = 0
    num_optimal = np.zeros(ITERS)
    bot_rewards = np.zeros(ITERS)
    print()
    for i in range(NUM_TESTS):
        print("\r", i, end="")
        rewards = np.random.randn(NUM_ACTIONS)
        opt = np.argmax(rewards)
        # bot = BanditLearner(actions, eps)
        bot = UCBBandit(actions, eps, 2)
        for t in range(ITERS):
            action = bot.action(t)
            reward = np.random.randn() + rewards[action]
            bot_rewards[t] += reward
            bot.update(action, reward, t)
            if action == opt:
                num_optimal[t] += 1
    print()
    print("Terminal optimal rate:", num_optimal[-1])
    sns.lineplot(range(ITERS), num_optimal / NUM_TESTS)
    plt.title("Fraction of bots making optimal choice")
    plt.ylim((0, 1))
    plt.show()


if __name__ == "__main__":
    argh.dispatch_command(test_bandits)
