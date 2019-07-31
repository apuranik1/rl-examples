"""Approximate N-step SARSA implementation for Mountain Car, parametrized by a small MLP.

Despite the simplicity of the setup, this has been quite finicky compared to any
of the tabular methods. The agent spends many early episodes with a terrible
value function approximation, and only after perhaps 10 of these does it reliably
start to learn a reasonable policy with the given parameters. The parameters currently
in the code were chosen as they gave consistent task completion within a few episodes.

Presumably learning would be sped up with complications like a replay buffer.
"""

from typing import Sequence

import numpy as np

import rl_examples.mountain_car as mc
from .neural import MLPEstimator
from .approximate_td import ApproximationTDNAgent


def mlp_mc_agent(
    env: mc.MountainCarEnvironment,
    mlp_hidden_shape: Sequence[int],
    mlp_lr: float = 0.01,
    epsilon: float = 0.01,
    n: int = 1,
    lr: float = 0.01,
    average: bool = False,
) -> ApproximationTDNAgent[mc.MountainCarState, mc.MountainCarAction]:
    featurizer = mc.MountainCarFeaturizer()
    estimator = MLPEstimator(
        featurizer.output_shape(), mlp_hidden_shape, "tanh", mlp_lr
    )
    return ApproximationTDNAgent(env, featurizer, estimator, epsilon, n, lr, average)


def run_example() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .psfa import train

    env = mc.MountainCarEnvironment()
    agent = mlp_mc_agent(env, [30], epsilon=0.0, n=20)

    def make_visual(state: mc.MountainCarState) -> None:
        fraction = (state.x + 1.2) / 1.7
        before = int(fraction * 30)
        print("-" * before + "#" + "-" * (29 - before))

    def graph_and_pause(t: int) -> None:
        print(f"Complete in {t} steps")
        MIN_X = mc.MountainCarEnvironment.MIN_X
        MAX_X = mc.MountainCarEnvironment.MAX_X
        MIN_VX = mc.MountainCarEnvironment.MIN_VX
        MAX_VX = mc.MountainCarEnvironment.MAX_VX
        x_vals = np.linspace(MIN_X, MAX_X, 20)
        vx_vals = np.linspace(MIN_VX, MAX_VX, 20)
        featurized = []
        for x in x_vals:
            for vx in vx_vals:
                state = mc.MountainCarState(x, vx, False)
                for a in mc.MountainCarAction:
                    featurized.append(agent._featurize(state, a))
        featurized_array = np.stack(featurized)
        values = agent.estimator.batch_estimate(featurized_array)
        values_shaped = values.reshape(
            len(x_vals), len(vx_vals), len(mc.MountainCarAction)
        )
        values = np.amax(values_shaped, axis=2)
        print(values.shape)
        sns.heatmap(values, xticklabels=x_vals, yticklabels=vx_vals)
        plt.savefig("values.png")
        plt.clf()
        # input("Press enter to continue training")
        # print()

    # train(env, agent, 10, on_action=lambda s, a, r, t: make_visual(s), on_episode_end=graph_and_pause)
    train(env, agent, 200, 3000, on_episode_end=graph_and_pause)


if __name__ == "__main__":
    run_example()
