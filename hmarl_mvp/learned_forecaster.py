"""Trainable queue-congestion forecaster using a small MLP.

The ``LearnedForecaster`` is trained on (current_queue_state → future_queue)
pairs collected from environment rollouts.  It replaces the heuristic
``MediumTermForecaster`` / ``ShortTermForecaster`` once trained.

Training flow:
1. Run rollout episodes with heuristic policies → collect queue traces
2. Build supervised dataset from queue traces
3. Train the MLP via ``train_forecaster(...)``
4. Plug the trained model into ``run_experiment(policy_type="learned_forecast")``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from .state import PortState

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


@dataclass
class ForecastDataset:
    """Supervised dataset of (input_features, target_queues) pairs.

    Each sample maps a snapshot of port state features to a future queue
    trajectory of length ``horizon``.

    Attributes
    ----------
    inputs:  shape ``(N, num_ports * input_features_per_port)``
    targets: shape ``(N, num_ports * horizon)``
    """

    inputs: np.ndarray
    targets: np.ndarray

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def to_tensors(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert *inputs* and *targets* to float32 tensors on *device*."""
        return (
            torch.as_tensor(self.inputs, dtype=torch.float32, device=device),
            torch.as_tensor(self.targets, dtype=torch.float32, device=device),
        )


def build_forecast_dataset(
    queue_traces: list[list[list[float]]],
    horizon: int,
    features_per_port: int = 3,
) -> ForecastDataset:
    """Build a supervised dataset from environment queue traces.

    Parameters
    ----------
    queue_traces:
        List of episodes, each episode is a list of per-step snapshots.
        Each snapshot is ``[queue_0, occupied_0, docks_0, queue_1, ...]``
        flattened across ports.  Length of each snapshot = num_ports * features_per_port.
    horizon:
        Number of future steps to predict.

    Returns
    -------
    ForecastDataset with input features and target queue values.
    """
    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for episode in queue_traces:
        T = len(episode)
        if T <= horizon:
            continue
        episode_arr = np.array(episode, dtype=np.float32)  # (T, num_ports * feat)
        num_features = episode_arr.shape[1]
        num_ports = num_features // features_per_port
        for t in range(T - horizon):
            inputs_list.append(episode_arr[t])
            # Target: queue values for next ``horizon`` steps (first feature per port)
            future = episode_arr[t + 1 : t + 1 + horizon]  # (horizon, num_features)
            # Extract queue column for each port (index 0, 3, 6, ...)
            queue_indices = np.arange(num_ports) * features_per_port
            target_queues = future[:, queue_indices]  # (horizon, num_ports)
            targets_list.append(target_queues.T.ravel())  # (num_ports * horizon,)

    if not inputs_list:
        empty_in = np.zeros((0, 1), dtype=np.float32)
        empty_tgt = np.zeros((0, 1), dtype=np.float32)
        return ForecastDataset(inputs=empty_in, targets=empty_tgt)

    return ForecastDataset(
        inputs=np.stack(inputs_list),
        targets=np.stack(targets_list),
    )


def collect_queue_traces(
    num_episodes: int,
    steps_per_episode: int,
    config: dict[str, Any],
    seed: int = 42,
) -> list[list[list[float]]]:
    """Run heuristic rollouts and collect per-step port state snapshots.

    Returns a list of episodes, each containing per-step flattened
    port state vectors ``[queue, occupied, docks]`` per port.
    """
    traces: list[list[list[float]]] = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        # Reconstruct per-step snapshots by running env with heuristic actions
        from .env import MaritimeEnv

        env = MaritimeEnv(config=config, seed=ep_seed)
        env.reset()
        episode_trace: list[list[float]] = []
        for _ in range(steps_per_episode):
            snapshot: list[float] = []
            for port in env.ports:
                snapshot.extend([float(port.queue), float(port.occupied), float(port.docks)])
            episode_trace.append(snapshot)
            actions = env.sample_stub_actions()
            env.step(actions)
        traces.append(episode_trace)
    return traces


# ---------------------------------------------------------------------------
# Learned forecaster model
# ---------------------------------------------------------------------------


class LearnedForecasterNet(nn.Module):
    """MLP that maps port state features → predicted queue trajectories."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.ReLU())  # queues are non-negative
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``(batch, input_dim) -> (batch, output_dim)``."""
        return self.net(x)


@dataclass
class LearnedForecaster:
    """Trained forecaster that can replace heuristic forecasters.

    Compatible with the ``MediumTermForecaster`` / ``ShortTermForecaster``
    interface: call ``.predict(ports, ...)`` to get ``(num_ports, horizon)``
    forecast arrays.
    """

    num_ports: int
    horizon: int
    features_per_port: int = 3
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    device: str = "cpu"

    _net: LearnedForecasterNet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        input_dim = self.num_ports * self.features_per_port
        output_dim = self.num_ports * self.horizon
        self._net = LearnedForecasterNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

    @property
    def net(self) -> LearnedForecasterNet:
        """Underlying PyTorch network module."""
        return self._net

    def predict(self, ports: list[PortState], rng: np.random.Generator | None = None) -> np.ndarray:
        """Predict future queue trajectories from current port state.

        Returns array of shape ``(num_ports, horizon)``.
        The ``rng`` argument is accepted for API compatibility but unused.
        """
        _ = rng
        features: list[float] = []
        for port in ports:
            features.extend([float(port.queue), float(port.occupied), float(port.docks)])
        x = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self._net(x)  # (1, num_ports * horizon)
        pred_np = pred.cpu().numpy().reshape(self.num_ports, self.horizon)
        return pred_np

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._net.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Training result container."""

    epoch_losses: list[float]
    final_loss: float
    num_samples: int


def train_forecaster(
    forecaster: LearnedForecaster,
    dataset: ForecastDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    verbose: bool = False,
) -> TrainResult:
    """Train the forecaster MLP on a supervised dataset.

    Returns
    -------
    TrainResult with per-epoch validation losses.
    """
    if len(dataset) == 0:
        return TrainResult(epoch_losses=[], final_loss=float("inf"), num_samples=0)

    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    indices = np.arange(n)
    np.random.default_rng(0).shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    device = forecaster.device
    inputs_t, targets_t = dataset.to_tensors(device)
    train_inputs, train_targets = inputs_t[train_idx], targets_t[train_idx]
    val_inputs, val_targets = inputs_t[val_idx], targets_t[val_idx]

    net = forecaster.net
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        net.train()
        perm = np.random.default_rng(epoch).permutation(n_train)
        epoch_train_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            batch_in = train_inputs[idx]
            batch_tgt = train_targets[idx]
            pred = net(batch_in)
            loss = loss_fn(pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1

        net.eval()
        with torch.no_grad():
            val_pred = net(val_inputs)
            val_loss = loss_fn(val_pred, val_targets).item()
        epoch_losses.append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_train = epoch_train_loss / max(n_batches, 1)
            print(f"  epoch {epoch:4d}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

    return TrainResult(
        epoch_losses=epoch_losses,
        final_loss=epoch_losses[-1] if epoch_losses else float("inf"),
        num_samples=n,
    )
