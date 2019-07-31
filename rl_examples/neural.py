from typing import Optional, Sequence

import mxnet as mx
import numpy as np
from mxnet.gluon import Block, Trainer, loss, nn

from rl_examples.approximation import TrainableEstimator


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    hidden_activation: Optional[str],
) -> Block:
    network = nn.HybridSequential()
    with network.name_scope():
        prev_size = input_dim
        for curr_size in hidden_dims:
            network.add(
                nn.Dense(curr_size, in_units=prev_size, activation=hidden_activation)
            )
            prev_size = curr_size
        # final layer never has activation
        network.add(nn.Dense(output_dim, in_units=prev_size))
    return network


class MLPEstimator(TrainableEstimator):
    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_dims: Sequence[int],
        hidden_activation: Optional[str],
        lr: float,
    ):
        input_dim, = input_shape  # must be 1D
        self.model = make_mlp(input_dim, hidden_dims, 1, hidden_activation)
        self.model.initialize()
        self.trainer = Trainer(self.model.collect_params(), mx.optimizer.Adam(lr))
        self.loss_fn = loss.L2Loss()

    def batch_estimate(self, data: np.ndarray) -> np.ndarray:
        tensor = mx.nd.array(data)
        result = self.model(tensor)
        return result.asnumpy()

    def batch_estimate_and_update(
        self, data: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        input_tensor = mx.nd.array(data)
        target_tensor = mx.nd.array(targets)
        with mx.autograd.record():
            result = self.model(input_tensor)
            loss = self.loss_fn(result, target_tensor)
        loss.backward()
        self.trainer.step(input_tensor.shape[0])
        return result.asnumpy()
