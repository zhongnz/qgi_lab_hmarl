"""Trainable queue-congestion forecasters: MLP baseline and GRU ablation.

Two architectures are provided for the April forecaster ablation study
(RQ3 — which forecast approach maximises decision quality?):

MLP  ``LearnedForecaster``
    Trained on single-snapshot → future-queue pairs.  Fast and simple.

GRU  ``RNNForecaster``
    GRU that processes a rolling window of port-state history.  Captures
    temporal dynamics that a snapshot-based MLP cannot see.

Training flow (both architectures):
1. Run rollout episodes with heuristic policies → collect queue traces.
2. Build supervised dataset (``build_forecast_dataset`` / ``build_rnn_dataset``).
3. Train via ``train_forecaster`` / ``train_rnn_forecaster``.
4. Plug into ``run_experiment(policy_type="learned_forecast")``.

For the ablation the three variants to compare are:
    heuristic  — ``MediumTermForecaster`` (no training)
    mlp        — ``LearnedForecaster``    (snapshot MLP)
    rnn        — ``RNNForecaster``        (GRU with queue-history window)
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


# ---------------------------------------------------------------------------
# Expanded feature collection (queue-history features for ablation)
# ---------------------------------------------------------------------------

def collect_expanded_queue_traces(
    num_episodes: int,
    steps_per_episode: int,
    config: dict[str, Any],
    seed: int = 42,
    features_per_port: int = 5,
) -> list[list[list[float]]]:
    """Run heuristic rollouts and collect per-step expanded port state snapshots.

    Compared to ``collect_queue_traces`` (3 features per port), this version
    captures 5 features per port:
    ``[queue, occupied, docks, cumulative_wait_hours_delta, vessels_served_delta]``

    The delta fields are per-step increments, normalised to avoid unbounded
    growth in cumulative accumulators.

    Returns a list of episodes, each containing per-step flattened state
    vectors of length ``num_ports * features_per_port``.
    """
    from .env import MaritimeEnv

    traces: list[list[list[float]]] = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = MaritimeEnv(config=config, seed=ep_seed)
        env.reset()
        episode_trace: list[list[float]] = []
        prev_wait = [0.0] * len(env.ports)
        prev_served = [0] * len(env.ports)
        for _ in range(steps_per_episode):
            snapshot: list[float] = []
            for i, port in enumerate(env.ports):
                wait_delta = port.cumulative_wait_hours - prev_wait[i]
                served_delta = float(port.vessels_served - prev_served[i])
                prev_wait[i] = port.cumulative_wait_hours
                prev_served[i] = port.vessels_served
                snapshot.extend([
                    float(port.queue),
                    float(port.occupied),
                    float(port.docks),
                    float(wait_delta),
                    float(served_delta),
                ])
            episode_trace.append(snapshot)
            actions = env.sample_stub_actions()
            env.step(actions)
        traces.append(episode_trace)
    return traces


# ---------------------------------------------------------------------------
# RNN dataset builder
# ---------------------------------------------------------------------------


@dataclass
class RNNForecastDataset:
    """Sequential supervised dataset for the GRU forecaster.

    Each sample is a window of ``seq_len`` consecutive port-state snapshots
    mapped to a future queue trajectory of length ``horizon``.

    Attributes
    ----------
    inputs_seq:
        shape ``(N, seq_len, input_dim)`` — sequence of port feature vectors.
    targets:
        shape ``(N, num_ports * horizon)`` — flattened future queue values.
    """

    inputs_seq: np.ndarray
    targets: np.ndarray

    def __len__(self) -> int:
        return self.inputs_seq.shape[0]

    def to_tensors(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert to float32 tensors on *device*."""
        return (
            torch.as_tensor(self.inputs_seq, dtype=torch.float32, device=device),
            torch.as_tensor(self.targets, dtype=torch.float32, device=device),
        )


def build_rnn_dataset(
    queue_traces: list[list[list[float]]],
    horizon: int,
    seq_len: int = 8,
    features_per_port: int = 3,
) -> RNNForecastDataset:
    """Build a windowed sequential dataset from environment queue traces.

    Parameters
    ----------
    queue_traces:
        List of episodes of per-step port-state snapshots (same format as
        ``build_forecast_dataset`` / ``collect_queue_traces``).
    horizon:
        Number of future steps to predict.
    seq_len:
        History window length fed to the GRU.

    Returns
    -------
    RNNForecastDataset with ``inputs_seq`` and ``targets``.
    """
    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for episode in queue_traces:
        T = len(episode)
        if T <= seq_len + horizon:
            continue
        episode_arr = np.array(episode, dtype=np.float32)  # (T, num_ports * feat)
        num_features = episode_arr.shape[1]
        num_ports = num_features // features_per_port

        for t in range(seq_len, T - horizon):
            # History window: (seq_len, input_dim)
            window = episode_arr[t - seq_len : t]
            inputs_list.append(window)
            # Target: queue values for next ``horizon`` steps
            future = episode_arr[t : t + horizon]  # (horizon, num_features)
            queue_indices = np.arange(num_ports) * features_per_port
            target_queues = future[:, queue_indices]  # (horizon, num_ports)
            targets_list.append(target_queues.T.ravel())  # (num_ports * horizon,)

    if not inputs_list:
        empty_in = np.zeros((0, seq_len, features_per_port), dtype=np.float32)
        empty_tgt = np.zeros((0, 1), dtype=np.float32)
        return RNNForecastDataset(inputs_seq=empty_in, targets=empty_tgt)

    return RNNForecastDataset(
        inputs_seq=np.stack(inputs_list),
        targets=np.stack(targets_list),
    )


# ---------------------------------------------------------------------------
# GRU-based forecaster model
# ---------------------------------------------------------------------------


class RNNForecasterNet(nn.Module):
    """GRU network: (batch, seq_len, input_dim) → (batch, output_dim).

    Architecture: stacked GRU → last hidden state → linear projection.
    Naturally captures temporal dependencies in port-queue dynamics that
    a snapshot MLP cannot see.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim),
            nn.ReLU(),  # queues are non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over sequence *x* of shape ``(batch, seq_len, input_dim)``."""
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]         # (batch, hidden_size) — top-layer final hidden
        return self.head(last_hidden)  # (batch, output_dim)


@dataclass
class RNNForecaster:
    """GRU-based trained forecaster with a rolling history buffer.

    Compatible with the ``LearnedForecaster`` / ``MediumTermForecaster``
    interface: call ``.predict(ports, rng)`` to get ``(num_ports, horizon)``
    forecast arrays.

    The history buffer accumulates the last ``seq_len`` port-state snapshots.
    When fewer than ``seq_len`` steps have been seen since the last reset the
    buffer is zero-padded at the front.

    Parameters
    ----------
    num_ports:    Number of ports in the environment.
    horizon:      Forecast horizon length (steps).
    seq_len:      History window fed to the GRU.
    features_per_port:
        Number of features per port in each snapshot (must match training).
    hidden_size / num_layers / dropout:
        GRU architecture hyper-parameters.
    device:       PyTorch device string (``"cpu"`` or ``"cuda"``).
    """

    num_ports: int
    horizon: int
    seq_len: int = 8
    features_per_port: int = 3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"

    _net: RNNForecasterNet = field(init=False, repr=False)
    _history: list[list[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        input_dim = self.num_ports * self.features_per_port
        output_dim = self.num_ports * self.horizon
        self._net = RNNForecasterNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self._history = []

    @property
    def net(self) -> RNNForecasterNet:
        """Underlying PyTorch GRU module."""
        return self._net

    def reset(self) -> None:
        """Clear the internal history buffer (call at each episode reset)."""
        self._history = []

    def predict(
        self, ports: list[PortState], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Predict future queue trajectories from current port state + history.

        Returns array of shape ``(num_ports, horizon)``.
        The ``rng`` argument is accepted for API compatibility but unused.
        """
        _ = rng
        # Build current snapshot
        snapshot: list[float] = []
        for port in ports:
            snapshot.extend([
                float(port.queue),
                float(port.occupied),
                float(port.docks),
            ])
        self._history.append(snapshot)
        # Keep only the last seq_len steps
        if len(self._history) > self.seq_len:
            self._history = self._history[-self.seq_len :]

        # Zero-pad if we don't have a full window yet
        input_dim = self.num_ports * self.features_per_port
        pad_len = self.seq_len - len(self._history)
        padding = [[0.0] * input_dim] * pad_len
        sequence = np.array(padding + self._history, dtype=np.float32)  # (seq_len, input_dim)

        x = torch.as_tensor(sequence[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self._net(x)  # (1, num_ports * horizon)
        return pred.cpu().numpy().reshape(self.num_ports, self.horizon)

    def save(self, path: str) -> None:
        """Save GRU model weights to disk."""
        torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load GRU model weights from disk."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._net.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# GRU training
# ---------------------------------------------------------------------------


def train_rnn_forecaster(
    forecaster: RNNForecaster,
    dataset: RNNForecastDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    verbose: bool = False,
) -> TrainResult:
    """Train the GRU forecaster on a sequential dataset.

    Identical interface/return type to ``train_forecaster`` so the two can
    be compared directly in the ablation study.
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
    train_in, train_tgt = inputs_t[train_idx], targets_t[train_idx]
    val_in, val_tgt = inputs_t[val_idx], targets_t[val_idx]

    net = forecaster.net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
            pred = net(train_in[idx])
            loss = loss_fn(pred, train_tgt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1

        net.eval()
        with torch.no_grad():
            val_pred = net(val_in)
            val_loss = loss_fn(val_pred, val_tgt).item()
        epoch_losses.append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_train = epoch_train_loss / max(n_batches, 1)
            print(f"  epoch {epoch:4d}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

    return TrainResult(
        epoch_losses=epoch_losses,
        final_loss=epoch_losses[-1] if epoch_losses else float("inf"),
        num_samples=n,
    )
