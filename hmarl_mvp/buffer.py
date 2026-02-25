"""Rollout buffer for collecting multi-agent experience trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Fixed-length rollout buffer for PPO-style on-policy training.

    Collects full rollouts of ``capacity`` transitions, then provides
    mini-batch iterators for policy updates.  One buffer instance is
    used per agent group (e.g. all vessels share one buffer).

    Stores numpy arrays internally and converts to tensors on demand
    to keep collection cheap and device-transfer explicit.
    """

    capacity: int
    obs_dim: int
    act_dim: int = 1
    gamma: float = 0.99
    lam: float = 0.95
    global_state_dim: int = 0

    # Internal storage (populated by ``add``).
    _obs: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)
    _log_probs: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _advantages: np.ndarray = field(init=False, repr=False)
    _returns: np.ndarray = field(init=False, repr=False)
    _global_states: np.ndarray = field(init=False, repr=False)
    _ptr: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear buffer and reset write pointer."""
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._log_probs = np.zeros(self.capacity, dtype=np.float32)
        self._values = np.zeros(self.capacity, dtype=np.float32)
        self._advantages = np.zeros(self.capacity, dtype=np.float32)
        self._returns = np.zeros(self.capacity, dtype=np.float32)
        gs_dim = max(self.global_state_dim, 1)
        self._global_states = np.zeros((self.capacity, gs_dim), dtype=np.float32)
        self._ptr = 0

    @property
    def size(self) -> int:
        """Number of transitions stored so far."""
        return self._ptr

    @property
    def full(self) -> bool:
        """True when the buffer has ``capacity`` transitions."""
        return self._ptr >= self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
        global_state: np.ndarray | None = None,
    ) -> None:
        """Append a single transition.  Raises if buffer is full."""
        if self._ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer is full — call reset() or compute_returns() first")
        self._obs[self._ptr] = np.asarray(obs, dtype=np.float32).ravel()[: self.obs_dim]
        act = np.atleast_1d(np.asarray(action, dtype=np.float32))
        self._actions[self._ptr] = act[: self.act_dim]
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._log_probs[self._ptr] = log_prob
        self._values[self._ptr] = value
        if global_state is not None and self.global_state_dim > 0:
            gs = np.asarray(global_state, dtype=np.float32).ravel()[: self.global_state_dim]
            self._global_states[self._ptr] = gs
        self._ptr += 1

    def set_reward(self, index: int, reward: float) -> None:
        """Overwrite the reward at *index* (supports negative indexing)."""
        if index < 0:
            index = self._ptr + index
        if index < 0 or index >= self._ptr:
            raise IndexError(f"Buffer index {index} out of range [0, {self._ptr})")
        self._rewards[index] = reward

    def set_done(self, index: int, done: float) -> None:
        """Overwrite the done flag at *index* (supports negative indexing)."""
        if index < 0:
            index = self._ptr + index
        if index < 0 or index >= self._ptr:
            raise IndexError(f"Buffer index {index} out of range [0, {self._ptr})")
        self._dones[index] = done

    def compute_returns(self, last_value: float = 0.0) -> None:
        """Compute GAE-Lambda advantages and discounted returns in-place."""
        n = self._ptr
        if n == 0:
            return
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self._dones[t])
            else:
                next_value = self._values[t + 1]
                next_non_terminal = 1.0 - float(self._dones[t])
            delta = (
                self._rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self._values[t]
            )
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            self._advantages[t] = gae
        self._returns[:n] = self._advantages[:n] + self._values[:n]

    def get_tensors(
        self,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Return all stored data as torch tensors on *device*."""
        n = self._ptr
        result = {
            "obs": torch.as_tensor(self._obs[:n], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self._actions[:n], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self._rewards[:n], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self._dones[:n], dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(self._log_probs[:n], dtype=torch.float32, device=device),
            "values": torch.as_tensor(self._values[:n], dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(
                self._advantages[:n], dtype=torch.float32, device=device
            ),
            "returns": torch.as_tensor(self._returns[:n], dtype=torch.float32, device=device),
        }
        if self.global_state_dim > 0:
            result["global_states"] = torch.as_tensor(
                self._global_states[:n], dtype=torch.float32, device=device
            )
        return result

    def minibatch_iter(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        shuffle: bool = True,
    ) -> Any:
        """Yield mini-batch dicts of tensors for PPO updates.

        The generator yields ``dict[str, Tensor]`` with keys matching
        ``get_tensors()``.
        """
        n = self._ptr
        if n == 0:
            return
        data = self.get_tensors(device)
        indices = np.arange(n)
        if shuffle:
            np.random.default_rng().shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {k: v[idx] for k, v in data.items()}


@dataclass
class MultiAgentRolloutBuffer:
    """Wrapper holding one ``RolloutBuffer`` per agent in a group.

    Typical usage: one ``MultiAgentRolloutBuffer`` for vessels, one for
    ports, one for coordinators.
    """

    num_agents: int
    capacity: int
    obs_dim: int
    act_dim: int = 1
    gamma: float = 0.99
    lam: float = 0.95
    global_state_dim: int = 0

    _buffers: list[RolloutBuffer] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buffers = [
            RolloutBuffer(
                capacity=self.capacity,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                gamma=self.gamma,
                lam=self.lam,
                global_state_dim=self.global_state_dim,
            )
            for _ in range(self.num_agents)
        ]

    def __getitem__(self, idx: int) -> RolloutBuffer:
        return self._buffers[idx]

    def reset(self) -> None:
        """Clear all agent buffers."""
        for buf in self._buffers:
            buf.reset()

    @property
    def full(self) -> bool:
        """True when all individual buffers are full."""
        return all(b.full for b in self._buffers)

    def compute_returns(self, last_values: list[float] | None = None) -> None:
        """Compute returns/advantages across all agent buffers."""
        last_values = last_values or [0.0] * self.num_agents
        for buf, lv in zip(self._buffers, last_values):
            buf.compute_returns(lv)
