"""Gymnasium-compatible wrapper around MaritimeEnv.

Provides a standard ``gymnasium.Env`` interface so the HMARL maritime
environment can be used with any Gymnasium-compatible RL library
(e.g. Stable-Baselines3, CleanRL, RLlib).

The wrapper flattens the multi-agent observation/action spaces into
a single-agent ``Box`` interface.  Agent-level structure is preserved
in the ``info`` dict for debugging.

Usage::

    import gymnasium as gym
    from hmarl_mvp.gym_wrapper import MaritimeGymEnv

    env = MaritimeGymEnv(config={"num_vessels": 4, "num_ports": 3})
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import MaritimeEnv
from .networks import obs_dim_from_env


class MaritimeGymEnv(gym.Env):
    """Gymnasium wrapper that presents the multi-agent env as single-agent.

    **Observation space**: ``Box`` — concatenation of all agent observations
    (coordinator + all vessels + all ports).

    **Action space**: ``Box`` — concatenation of all continuous agent actions:
    - Coordinator: ``[dest_port, departure_window_hours, emission_budget]``
    - Per vessel: ``[target_speed, request_arrival_slot]``
    - Per port: ``[service_rate, accept_requests]``

    Parameters
    ----------
    config:
        Environment config overrides passed to ``MaritimeEnv``.
    seed:
        Random seed.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = dict(config or {})
        self._seed = seed
        self._env = MaritimeEnv(config=self._config, seed=self._seed)

        cfg = self._env.cfg
        self._num_vessels = cfg["num_vessels"]
        self._num_ports = cfg["num_ports"]

        # Compute observation dimensions
        obs_dims = obs_dim_from_env(cfg)
        self._obs_dim = (
            obs_dims["coordinator"]
            + self._num_vessels * obs_dims["vessel"]
            + self._num_ports * obs_dims["port"]
        )
        self._vessel_obs_dim = obs_dims["vessel"]
        self._port_obs_dim = obs_dims["port"]
        self._coord_obs_dim = obs_dims["coordinator"]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Action dimensions
        coord_action_dim = 3  # dest_port, departure_window, emission_budget
        vessel_action_dim = 2  # target_speed, request_arrival_slot
        port_action_dim = 2  # service_rate, accept_requests

        self._coord_action_dim = coord_action_dim
        self._vessel_action_dim = vessel_action_dim
        self._port_action_dim = port_action_dim
        total_action_dim = (
            coord_action_dim
            + self._num_vessels * vessel_action_dim
            + self._num_ports * port_action_dim
        )
        # Action bounds
        low_parts = [
            np.array([0.0, 0.0, 0.0]),  # coordinator
        ]
        high_parts = [
            np.array([float(self._num_ports - 1), 48.0, 100.0]),  # coordinator
        ]
        for _ in range(self._num_vessels):
            low_parts.append(np.array([cfg["speed_min"], 0.0]))
            high_parts.append(np.array([cfg["speed_max"], 1.0]))
        for _ in range(self._num_ports):
            low_parts.append(np.array([0.0, 0.0]))
            high_parts.append(np.array([10.0, 10.0]))

        self.action_space = spaces.Box(
            low=np.concatenate(low_parts).astype(np.float32),
            high=np.concatenate(high_parts).astype(np.float32),
            shape=(total_action_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return (observation, info)."""
        if seed is not None:
            self._env.seed = seed
        raw_obs = self._env.reset()
        flat_obs = self._flatten_obs(raw_obs)
        return flat_obs, {"raw_obs": raw_obs}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step and return Gymnasium 5-tuple."""
        actions_dict = self._unflatten_action(action)
        raw_obs, raw_rewards, done, info = self._env.step(actions_dict)
        flat_obs = self._flatten_obs(raw_obs)

        # Aggregate reward: sum of vessel + port + coordinator rewards
        vessel_reward = float(sum(raw_rewards.get("vessels", [])))
        port_reward = float(sum(raw_rewards.get("ports", [])))
        coord_reward = float(raw_rewards.get("coordinator", 0.0))
        total_reward = vessel_reward + port_reward + coord_reward

        info["raw_obs"] = raw_obs
        info["raw_rewards"] = raw_rewards
        info["vessel_reward"] = vessel_reward
        info["port_reward"] = port_reward
        info["coordinator_reward"] = coord_reward
        if self._env._weather is not None:
            info["weather_matrix"] = self._env._weather.copy()

        # Gymnasium convention: terminated vs truncated
        terminated = done
        truncated = False

        return flat_obs, total_reward, terminated, truncated, info

    def _flatten_obs(self, obs: dict[str, Any]) -> np.ndarray:
        """Concatenate structured observations into a flat vector."""
        parts: list[np.ndarray] = []

        # Coordinator obs (first coordinator, already padded)
        coord = np.asarray(obs["coordinator"], dtype=np.float32)
        parts.append(coord[:self._coord_obs_dim])

        # Vessel observations
        for v_obs in obs.get("vessels", []):
            parts.append(np.asarray(v_obs, dtype=np.float32))

        # Port observations
        for p_obs in obs.get("ports", []):
            parts.append(np.asarray(p_obs, dtype=np.float32))

        flat = np.concatenate(parts)
        # Pad/truncate to expected dimension
        if len(flat) < self._obs_dim:
            flat = np.concatenate([flat, np.zeros(self._obs_dim - len(flat), dtype=np.float32)])
        return flat[:self._obs_dim]

    def _unflatten_action(self, action: np.ndarray) -> dict[str, Any]:
        """Convert flat action vector back to structured action dict."""
        action = np.asarray(action, dtype=np.float32)
        idx = 0

        # Coordinator action
        coord_raw = action[idx:idx + self._coord_action_dim]
        idx += self._coord_action_dim
        coordinator_action = {
            "dest_port": int(np.clip(np.round(coord_raw[0]), 0, self._num_ports - 1)),
            "departure_window_hours": float(max(coord_raw[1], 0)),
            "emission_budget": float(max(coord_raw[2], 0)),
        }

        # Vessel actions
        vessel_actions = []
        for _ in range(self._num_vessels):
            v_raw = action[idx:idx + self._vessel_action_dim]
            idx += self._vessel_action_dim
            vessel_actions.append({
                "target_speed": float(v_raw[0]),
                "request_arrival_slot": bool(v_raw[1] > 0.5),
            })

        # Port actions
        port_actions = []
        for _ in range(self._num_ports):
            p_raw = action[idx:idx + self._port_action_dim]
            idx += self._port_action_dim
            port_actions.append({
                "service_rate": int(max(np.round(p_raw[0]), 0)),
                "accept_requests": int(max(np.round(p_raw[1]), 0)),
            })

        return {
            "coordinator": coordinator_action,
            "coordinators": [coordinator_action],
            "vessels": vessel_actions,
            "ports": port_actions,
        }

    @property
    def unwrapped(self) -> MaritimeEnv:  # type: ignore[override]
        """Return the underlying MaritimeEnv."""
        return self._env
