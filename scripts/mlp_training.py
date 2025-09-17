#!/usr/bin/env python3
"""Simulate a CPU-only training job for environments with scheduled resources.

The script mimics the console output of a deep learning training loop without
actually requiring GPUs.  It performs small NumPy computations on the CPU so
that the job scheduler sees real activity, while staying lightweight.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

# Make sure no CUDA devices are visible to downstream libraries even if the
# host machine has GPUs available.  This must be set before torch/tensorflow
# (if they were ever imported) to guarantee CPU-only execution.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass
class TrainingConfig:
    epochs: int = 5
    steps_per_epoch: int = 50
    batch_size: int = 32
    input_dim: int = 256
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    sleep: float = 0.05


@dataclass
class TrainingState:
    weights: np.ndarray
    bias: np.ndarray

    @staticmethod
    def initialize(input_dim: int, hidden_dim: int) -> "TrainingState":
        rng = np.random.default_rng(seed=42)
        weights = rng.normal(scale=0.02, size=(input_dim, hidden_dim))
        bias = np.zeros(hidden_dim)
        return TrainingState(weights=weights, bias=bias)


class DummyDataLoader:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=1234)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for _ in range(self.cfg.steps_per_epoch):
            inputs = self.rng.normal(size=(self.cfg.batch_size, self.cfg.input_dim))
            targets = self.rng.normal(size=(self.cfg.batch_size, self.cfg.hidden_dim))
            yield inputs, targets


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def run_epoch(cfg: TrainingConfig, state: TrainingState, epoch: int) -> float:
    data_loader = DummyDataLoader(cfg)
    losses = []

    for step, (inputs, targets) in enumerate(data_loader, start=1):
        predictions = np.tanh(inputs @ state.weights + state.bias)
        loss = mse_loss(predictions, targets)

        # Simple gradient-like update to make the loss fluctuate but trend down.
        grad_w = (inputs.T @ (predictions - targets)) / cfg.batch_size
        grad_b = np.mean(predictions - targets, axis=0)
        state.weights -= cfg.learning_rate * grad_w
        state.bias -= cfg.learning_rate * grad_b

        # Introduce a small pause so job schedulers see sustained CPU usage.
        time.sleep(cfg.sleep)

        losses.append(loss)
        if step % max(1, cfg.steps_per_epoch // 5) == 0 or step == cfg.steps_per_epoch:
            avg_loss = sum(losses[-5:]) / min(5, len(losses))
            print(
                f"Epoch {epoch:02d} | Step {step:03d}/{cfg.steps_per_epoch:03d} "
                f"| Loss: {avg_loss:.4f}"
            )

    return float(np.mean(losses))


def maybe_save_checkpoint(epoch: int, loss: float) -> None:
    # In a real training job you would persist model parameters here.  We only
    # simulate the behavior with a message so that downstream monitoring sees
    # progress updates.
    if epoch % 2 == 0:
        print(f"Checkpoint saved for epoch {epoch:02d} (loss={loss:.4f})")


def train(cfg: TrainingConfig) -> None:
    print("Starting dummy CPU training job...")
    state = TrainingState.initialize(cfg.input_dim, cfg.hidden_dim)
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        loss = run_epoch(cfg, state, epoch)
        duration = time.time() - epoch_start

        if loss < best_loss:
            best_loss = loss
            print(f"New best loss {best_loss:.4f} achieved at epoch {epoch:02d}")

        maybe_save_checkpoint(epoch, loss)
        print(f"Epoch {epoch:02d} completed in {duration:.1f}s\n")

    print("Training finished. Best loss: {:.4f}".format(best_loss))


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=TrainingConfig.steps_per_epoch
    )
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--input-dim", type=int, default=TrainingConfig.input_dim)
    parser.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument(
        "--sleep",
        type=float,
        default=TrainingConfig.sleep,
        help="Artificial delay (seconds) between steps to simulate work",
    )
    return TrainingConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
