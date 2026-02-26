"""Multi-Agent PPO (MAPPO) trainer with Centralised Training Decentralised Execution.

This module connects the existing infrastructure — ``MaritimeEnv``,
``ActorCritic`` networks, and ``MultiAgentRolloutBuffer`` — into a
complete on-policy training loop.

Design decisions
----------------
* **CTDE**: actors use local observations; the shared critic receives
  ``env.get_global_state()``.
* **Parameter sharing**: all vessels share a single ``ActorCritic``, all
  ports share one, and the coordinator(s) share one.
* Action translation:
  - Vessel (continuous, dim=1): NN output → clamp to [speed_min, speed_max].
  - Port (discrete, dim=docks+1): NN output index → ``service_rate``.
  - Coordinator (discrete, dim=num_ports): NN output index → ``dest_port``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .buffer import MultiAgentRolloutBuffer
from .env import MaritimeEnv
from .networks import ActorCritic, build_actor_critics, obs_dim_from_env

# ---------------------------------------------------------------------------
# Global-state dimension helper
# ---------------------------------------------------------------------------


def global_state_dim_from_config(config: dict[str, Any]) -> int:
    """Compute the flattened global-state dimension for the CTDE critic.

    Mirrors the concatenation order in ``MaritimeEnv.get_global_state()``.
    Each coordinator's observation is **zero-padded** to the maximum
    coordinator dimension so the global state has a deterministic size::

        N_c * (num_ports * medium_d + num_vessels * 4 + 1)
        + num_vessels * vessel_dim + num_ports * port_dim
        + num_ports + 1
    """
    dims = obs_dim_from_env(config)
    num_vessels = int(config["num_vessels"])
    num_ports = int(config["num_ports"])
    num_coordinators = int(config.get("num_coordinators", 1))
    return (
        num_coordinators * dims["coordinator"]  # padded to max coordinator dim
        + num_vessels * dims["vessel"]
        + num_ports * dims["port"]
        + num_ports  # global_congestion
        + 1  # total_emissions
    )


# ---------------------------------------------------------------------------
# Running statistics for reward normalisation
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Welford online mean/variance tracker for reward normalisation.

    Tracks running statistics and provides ``normalize`` that centres and
    scales a value by the tracked mean and standard deviation.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon
        self._epsilon = epsilon

    def update(self, x: float) -> None:
        """Incorporate a single scalar observation."""
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def update_batch(self, xs: list[float] | np.ndarray) -> None:
        """Incorporate a batch of scalar observations."""
        for x in xs:
            self.update(float(x))

    def normalize(self, x: float) -> float:
        """Return ``(x - mean) / std``."""
        std = max(np.sqrt(max(self.var, 0.0)), self._epsilon)
        return (x - self.mean) / std

    @property
    def std(self) -> float:
        return float(max(np.sqrt(max(self.var, 0.0)), self._epsilon))


class ObsRunningMeanStd:
    """Per-dimension Welford running mean/variance for observation normalisation.

    Maintains a running mean and variance vector so each feature of an
    observation is independently normalised to zero-mean, unit-variance.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8) -> None:
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count: float = epsilon
        self._epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Incorporate a single observation vector."""
        x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def update_batch(self, xs: np.ndarray) -> None:
        """Incorporate a batch of observation vectors (N × dim)."""
        for x in xs:
            self.update(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return ``(x - mean) / std`` element-wise."""
        std = np.sqrt(np.maximum(self.var, 0.0)).clip(min=self._epsilon)
        return (np.asarray(x, dtype=np.float64) - self.mean) / std


# ---------------------------------------------------------------------------
# PPO update result
# ---------------------------------------------------------------------------


@dataclass
class PPOUpdateResult:
    """Metrics from a single PPO update pass."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    weight_norm: float = 0.0
    approx_kl: float = 0.0
    entropy_coeff: float = 0.0
    kl_early_stopped: bool = False


# ---------------------------------------------------------------------------
# Action translation helpers
# ---------------------------------------------------------------------------


def _nn_to_vessel_action(
    raw: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Convert a 1-D continuous NN output to a vessel action dict."""
    speed = float(raw.detach().cpu().item())
    speed = max(config["speed_min"], min(config["speed_max"], speed))
    return {"target_speed": speed, "request_arrival_slot": True}


def _nn_to_port_action(
    raw: torch.Tensor,
    port_idx: int,
    env: MaritimeEnv,
) -> dict[str, Any]:
    """Convert a discrete NN output to a port action dict."""
    service_rate = int(raw.detach().cpu().item())
    port = env.ports[port_idx]
    available = max(port.docks - port.occupied, 0)
    return {"service_rate": service_rate, "accept_requests": available}


def _nn_to_coordinator_action(
    raw: torch.Tensor,
    coordinator_idx: int,
    env: MaritimeEnv,
    assignments: dict[int, list[int]],
) -> dict[str, Any]:
    """Convert a discrete NN output to a coordinator action dict.

    The selected port is the *primary* destination.  Assigned vessels are
    distributed across ports in order of proximity to the primary port so
    each vessel receives a unique (or near-unique) destination rather than
    all being sent to the same place — matching the per-vessel routing that
    heuristic policies already perform.
    """
    dest_port = int(raw.detach().cpu().item())
    local_ids = assignments.get(coordinator_idx, [])
    num_ports = env.num_ports

    # Build port ordering: primary first, then by ascending distance
    distances = env.distance_nm[dest_port]
    port_order = sorted(range(num_ports), key=lambda p: (distances[p], p))

    per_vessel_dest: dict[int, int] = {}
    for i, vid in enumerate(local_ids):
        per_vessel_dest[vid] = port_order[i % len(port_order)] if port_order else dest_port

    total_emissions = sum(v.emissions for v in env.vessels)
    return {
        "dest_port": dest_port,
        "per_vessel_dest": per_vessel_dest,
        "departure_window_hours": 12,
        "emission_budget": max(50.0 - total_emissions * 0.1, 10.0),
    }


# ---------------------------------------------------------------------------
# MAPPO Trainer
# ---------------------------------------------------------------------------


@dataclass
class MAPPOConfig:
    """Hyper-parameters for MAPPO training."""

    # Rollout / environment
    rollout_length: int = 64
    num_epochs: int = 4
    minibatch_size: int = 32
    # PPO
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    # Optimiser
    lr: float = 3e-4
    lr_end: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # KL early stopping (set target_kl=0 to disable)
    target_kl: float = 0.02
    # Entropy scheduling (linear decay from entropy_coeff → entropy_coeff_end)
    entropy_coeff_end: float | None = None
    # Training schedule
    total_iterations: int = 0
    # Misc
    device: str = "cpu"
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    normalize_rewards: bool = True
    normalize_observations: bool = True


class MAPPOTrainer:
    """End-to-end MAPPO trainer for the maritime HMARL environment.

    Usage::

        trainer = MAPPOTrainer(env_config={...})
        for iteration in range(num_iterations):
            rollout_info = trainer.collect_rollout()
            update_info = trainer.update()
    """

    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        mappo_config: MAPPOConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.mappo_cfg = mappo_config or MAPPOConfig()
        self.device = torch.device(self.mappo_cfg.device)

        # Build environment
        env_config = dict(env_config or {})
        env_config.setdefault("rollout_steps", self.mappo_cfg.rollout_length + 5)
        self.env = MaritimeEnv(config=env_config, seed=seed)
        self.cfg = self.env.cfg

        # Observation / action dimensions
        self.obs_dims = obs_dim_from_env(self.cfg)
        self.global_dim = global_state_dim_from_config(self.cfg)

        # Actor-critic networks (parameter-shared per agent type)
        self.actor_critics: dict[str, ActorCritic] = build_actor_critics(
            config=self.cfg,
            vessel_obs_dim=self.obs_dims["vessel"],
            port_obs_dim=self.obs_dims["port"],
            coordinator_obs_dim=self.obs_dims["coordinator"],
            global_state_dim=self.global_dim,
            hidden_dims=self.mappo_cfg.hidden_dims,
        )
        for ac in self.actor_critics.values():
            ac.to(self.device)

        # Optimisers (one per agent type)
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            name: torch.optim.Adam(ac.parameters(), lr=self.mappo_cfg.lr)
            for name, ac in self.actor_critics.items()
        }

        # Rollout buffers
        self._build_buffers()

        # Reward normalisers (one per agent type)
        self._reward_normalizers: dict[str, RunningMeanStd] = {
            "vessel": RunningMeanStd(),
            "port": RunningMeanStd(),
            "coordinator": RunningMeanStd(),
        }

        # Observation normalisers — per-dimension running stats
        self._obs_normalizers: dict[str, ObsRunningMeanStd] = {
            "vessel": ObsRunningMeanStd(self.obs_dims["vessel"]),
            "port": ObsRunningMeanStd(self.obs_dims["port"]),
            "coordinator": ObsRunningMeanStd(self.obs_dims["coordinator"]),
        }

        # Iteration counter for LR / entropy scheduling
        self._iteration: int = 0

        # Episode bookkeeping
        self._episode_rewards: list[float] = []
        self._obs: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Learning-rate scheduling
    # ------------------------------------------------------------------

    def _step_lr(self) -> None:
        """Apply linear LR annealing if ``total_iterations`` is configured."""
        self._iteration += 1
        total = self.mappo_cfg.total_iterations
        if total <= 0:
            return
        frac = max(1.0 - self._iteration / total, 0.0)
        lr = self.mappo_cfg.lr_end + frac * (self.mappo_cfg.lr - self.mappo_cfg.lr_end)
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        """Return the current learning rate from the first optimiser."""
        opt = next(iter(self.optimizers.values()))
        return float(opt.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # Entropy coefficient scheduling
    # ------------------------------------------------------------------

    @property
    def current_entropy_coeff(self) -> float:
        """Return the current entropy coefficient after linear decay.

        If ``entropy_coeff_end`` is *None* (default) the entropy
        coefficient stays constant.  Otherwise it linearly decays
        from ``entropy_coeff`` to ``entropy_coeff_end`` over
        ``total_iterations``.
        """
        cfg = self.mappo_cfg
        if cfg.entropy_coeff_end is None:
            return cfg.entropy_coeff
        total = cfg.total_iterations
        if total <= 0:
            return cfg.entropy_coeff
        frac = max(1.0 - self._iteration / total, 0.0)
        return cfg.entropy_coeff_end + frac * (cfg.entropy_coeff - cfg.entropy_coeff_end)

    # ------------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs: np.ndarray, agent_type: str) -> np.ndarray:
        """Normalize an observation vector using running statistics.

        Updates the running statistics and returns the normalised vector.
        If observation normalization is disabled, returns the input unchanged.
        """
        norm = self._obs_normalizers[agent_type]
        norm.update(obs)
        if not self.mappo_cfg.normalize_observations:
            return obs
        return norm.normalize(obs).astype(np.float32)

    def _eval_normalize_obs(self, obs: np.ndarray, agent_type: str) -> np.ndarray:
        """Normalize an observation using existing stats (no update).

        Used during evaluation to apply the same normalization learned
        during training without contaminating the running statistics.
        """
        if not self.mappo_cfg.normalize_observations:
            return obs
        return self._obs_normalizers[agent_type].normalize(obs).astype(np.float32)

    # ------------------------------------------------------------------
    # Buffer setup
    # ------------------------------------------------------------------

    def _build_buffers(self) -> None:
        """Create multi-agent rollout buffers for each agent type."""
        rl = self.mappo_cfg.rollout_length
        g = self.mappo_cfg.gamma
        lam = self.mappo_cfg.gae_lambda

        self.vessel_buf = MultiAgentRolloutBuffer(
            num_agents=self.env.num_vessels,
            capacity=rl,
            obs_dim=self.obs_dims["vessel"],
            act_dim=1,
            gamma=g,
            lam=lam,
            global_state_dim=self.global_dim,
        )
        self.port_buf = MultiAgentRolloutBuffer(
            num_agents=self.env.num_ports,
            capacity=rl,
            obs_dim=self.obs_dims["port"],
            act_dim=1,  # store discrete as scalar
            gamma=g,
            lam=lam,
            global_state_dim=self.global_dim,
        )
        self.coordinator_buf = MultiAgentRolloutBuffer(
            num_agents=self.env.num_coordinators,
            capacity=rl,
            obs_dim=self.obs_dims["coordinator"],
            act_dim=1,  # store discrete as scalar
            gamma=g,
            lam=lam,
            global_state_dim=self.global_dim,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict[str, float]:
        """Collect one rollout of ``rollout_length`` steps.

        Returns a dict with summary statistics (mean rewards, etc.).
        """
        self._reset_buffers()
        obs = self.env.reset()
        self._obs = obs
        total_reward = 0.0

        vessel_ac = self.actor_critics["vessel"]
        port_ac = self.actor_critics["port"]
        coord_ac = self.actor_critics["coordinator"]

        vessel_ac.eval()
        port_ac.eval()
        coord_ac.eval()

        for _step in range(self.mappo_cfg.rollout_length):
            global_state = self.env.get_global_state()
            gs_np = np.array(global_state, dtype=np.float32)
            gs_tensor = torch.as_tensor(
                global_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # --- Vessel actions ---
            vessel_actions_list: list[dict[str, Any]] = []
            with torch.no_grad():
                for i, v_obs in enumerate(obs["vessels"]):
                    v_obs_n = self._normalize_obs(v_obs, "vessel")
                    v_obs_t = torch.as_tensor(
                        v_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    action_t, log_prob_t, value_t = vessel_ac.get_action_and_value(
                        v_obs_t, gs_tensor
                    )
                    action_dict = _nn_to_vessel_action(action_t.squeeze(0), self.cfg)
                    vessel_actions_list.append(action_dict)
                    self.vessel_buf[i].add(
                        obs=v_obs_n,
                        action=action_t.squeeze(0).cpu().numpy(),
                        reward=0.0,  # filled after step
                        done=False,
                        log_prob=log_prob_t.item(),
                        value=value_t.item(),
                        global_state=gs_np,
                    )

            # --- Port actions ---
            port_actions_list: list[dict[str, Any]] = []
            with torch.no_grad():
                for i, p_obs in enumerate(obs["ports"]):
                    p_obs_n = self._normalize_obs(p_obs, "port")
                    p_obs_t = torch.as_tensor(
                        p_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    action_t, log_prob_t, value_t = port_ac.get_action_and_value(
                        p_obs_t, gs_tensor
                    )
                    action_dict = _nn_to_port_action(action_t.squeeze(0), i, self.env)
                    port_actions_list.append(action_dict)
                    self.port_buf[i].add(
                        obs=p_obs_n,
                        action=np.array([action_t.squeeze(0).cpu().item()]),
                        reward=0.0,
                        done=False,
                        log_prob=log_prob_t.item(),
                        value=value_t.item(),
                        global_state=gs_np,
                    )

            # --- Coordinator actions ---
            assignments = self.env._build_assignments()
            coordinator_actions_list: list[dict[str, Any]] = []
            with torch.no_grad():
                for i, c_obs in enumerate(obs["coordinators"]):
                    c_obs_n = self._normalize_obs(c_obs, "coordinator")
                    c_obs_t = torch.as_tensor(
                        c_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    action_t, log_prob_t, value_t = coord_ac.get_action_and_value(
                        c_obs_t, gs_tensor
                    )
                    action_dict = _nn_to_coordinator_action(
                        action_t.squeeze(0), i, self.env, assignments
                    )
                    coordinator_actions_list.append(action_dict)
                    self.coordinator_buf[i].add(
                        obs=c_obs_n,
                        action=np.array([action_t.squeeze(0).cpu().item()]),
                        reward=0.0,
                        done=False,
                        log_prob=log_prob_t.item(),
                        value=value_t.item(),
                        global_state=gs_np,
                    )

            # --- Step environment ---
            actions_dict: dict[str, Any] = {
                "coordinator": coordinator_actions_list[0] if coordinator_actions_list else {},
                "coordinators": coordinator_actions_list,
                "vessels": vessel_actions_list,
                "ports": port_actions_list,
            }
            obs, rewards, done, _info = self.env.step(actions_dict)
            self._obs = obs

            # --- Write rewards back into buffers ---
            norm = self.mappo_cfg.normalize_rewards
            for i, r in enumerate(rewards["vessels"]):
                raw_r = float(r)
                self._reward_normalizers["vessel"].update(raw_r)
                rew = self._reward_normalizers["vessel"].normalize(raw_r) if norm else raw_r
                buf = self.vessel_buf[i]
                buf.set_reward(-1, rew)
                buf.set_done(-1, float(done))

            for i, r in enumerate(rewards["ports"]):
                raw_r = float(r)
                self._reward_normalizers["port"].update(raw_r)
                rew = self._reward_normalizers["port"].normalize(raw_r) if norm else raw_r
                buf = self.port_buf[i]
                buf.set_reward(-1, rew)
                buf.set_done(-1, float(done))

            for i, r in enumerate(rewards["coordinators"]):
                raw_r = float(r)
                self._reward_normalizers["coordinator"].update(raw_r)
                rew = self._reward_normalizers["coordinator"].normalize(raw_r) if norm else raw_r
                buf = self.coordinator_buf[i]
                buf.set_reward(-1, rew)
                buf.set_done(-1, float(done))

            step_reward = float(np.mean(rewards["vessels"])) + float(rewards["coordinator"])
            total_reward += step_reward

            if done:
                obs = self.env.reset()
                self._obs = obs

        # --- Compute last values for GAE ---
        with torch.no_grad():
            global_state = self.env.get_global_state()
            gs_tensor = torch.as_tensor(
                global_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            vessel_last = []
            for _v_obs in obs["vessels"]:
                vessel_last.append(vessel_ac.critic(gs_tensor).item())
            self.vessel_buf.compute_returns(vessel_last)

            port_last = []
            for _p_obs in obs["ports"]:
                port_last.append(port_ac.critic(gs_tensor).item())
            self.port_buf.compute_returns(port_last)

            coord_last = []
            for _c_obs in obs["coordinators"]:
                coord_last.append(coord_ac.critic(gs_tensor).item())
            self.coordinator_buf.compute_returns(coord_last)

        mean_reward = total_reward / self.mappo_cfg.rollout_length
        self._episode_rewards.append(mean_reward)
        return {"mean_reward": mean_reward, "total_reward": total_reward}

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, PPOUpdateResult]:
        """Run PPO update epochs on the collected rollout.

        Returns per-agent-type ``PPOUpdateResult``.
        """
        results: dict[str, PPOUpdateResult] = {}
        for agent_type, ac in self.actor_critics.items():
            buf = self._get_buffer(agent_type)
            result = self._ppo_update(ac, buf, self.optimizers[agent_type])
            results[agent_type] = result
        # Step LR scheduler after all agent types are updated
        self._step_lr()
        return results

    def _ppo_update(
        self,
        ac: ActorCritic,
        multi_buf: MultiAgentRolloutBuffer,
        optimizer: torch.optim.Optimizer,
    ) -> PPOUpdateResult:
        """PPO clipped surrogate update for one agent type."""
        ac.train()
        cfg = self.mappo_cfg

        # Collect all agent data into flat tensors
        all_obs: list[torch.Tensor] = []
        all_global: list[torch.Tensor] = []
        all_actions: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_advantages: list[torch.Tensor] = []
        all_returns: list[torch.Tensor] = []

        # Fallback global state when buffer has no per-step data
        fallback_gs = self.env.get_global_state()

        for i in range(multi_buf.num_agents):
            data = multi_buf[i].get_tensors(self.device)
            n = data["obs"].shape[0]
            if n == 0:
                continue
            all_obs.append(data["obs"])
            # Use per-step global states stored during rollout collection
            if "global_states" in data:
                all_global.append(data["global_states"])
            else:
                gs_t = torch.as_tensor(
                    fallback_gs, dtype=torch.float32, device=self.device
                ).unsqueeze(0).expand(n, -1)
                all_global.append(gs_t)
            all_actions.append(data["actions"])
            all_log_probs.append(data["log_probs"])
            all_values.append(data["values"])
            all_advantages.append(data["advantages"])
            all_returns.append(data["returns"])

        if not all_obs:
            return PPOUpdateResult()

        obs_t = torch.cat(all_obs)
        global_t = torch.cat(all_global)
        actions_t = torch.cat(all_actions)
        old_log_probs_t = torch.cat(all_log_probs)
        old_values_t = torch.cat(all_values)
        advantages_t = torch.cat(all_advantages)
        returns_t = torch.cat(all_returns)

        # Normalise advantages
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_n = obs_t.shape[0]
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_frac = 0.0
        epoch_grad_norm = 0.0
        epoch_approx_kl = 0.0
        num_updates = 0
        kl_early_stopped = False

        ent_coeff = self.current_entropy_coeff

        for _epoch in range(cfg.num_epochs):
            if kl_early_stopped:
                break
            perm = torch.randperm(total_n, device=self.device)
            for start in range(0, total_n, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, total_n)
                idx = perm[start:end]

                mb_obs = obs_t[idx]
                mb_global = global_t[idx]
                mb_actions = actions_t[idx]
                mb_old_lp = old_log_probs_t[idx]
                mb_old_val = old_values_t[idx]
                mb_adv = advantages_t[idx]
                mb_ret = returns_t[idx]

                new_lp, entropy, values = ac.evaluate(mb_obs, mb_global, mb_actions)

                # Clipped surrogate objective
                ratio = (new_lp - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss (PPO2-style)
                v_clip = cfg.value_clip_eps
                if v_clip > 0.0:
                    values_clipped = mb_old_val + (values - mb_old_val).clamp(
                        -v_clip, v_clip
                    )
                    vf_loss_unclipped = (values - mb_ret) ** 2
                    vf_loss_clipped = (values_clipped - mb_ret) ** 2
                    value_loss = 0.5 * torch.max(
                        vf_loss_unclipped, vf_loss_clipped
                    ).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, mb_ret)

                # Entropy bonus (uses scheduled coefficient)
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coeff * value_loss
                    - ent_coeff * entropy_mean
                )

                optimizer.zero_grad()
                loss.backward()

                # Capture pre-clip gradient norm for diagnostics
                grad_norm_t = nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                    # Approximate KL divergence: mean((ratio - 1) - log(ratio))
                    log_ratio = new_lp - mb_old_lp
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy_mean.item()
                epoch_clip_frac += clip_frac
                epoch_grad_norm += float(grad_norm_t)
                epoch_approx_kl += approx_kl
                num_updates += 1

            # KL early stopping: check after each full epoch
            if cfg.target_kl > 0 and num_updates > 0:
                mean_kl = epoch_approx_kl / num_updates
                if mean_kl > cfg.target_kl:
                    kl_early_stopped = True

        # Compute weight norm across all parameters
        weight_norm = float(
            sum(p.data.norm().item() ** 2 for p in ac.parameters()) ** 0.5
        )

        denom = max(num_updates, 1)
        return PPOUpdateResult(
            policy_loss=epoch_policy_loss / denom,
            value_loss=epoch_value_loss / denom,
            entropy=epoch_entropy / denom,
            total_loss=(epoch_policy_loss + epoch_value_loss) / denom,
            clip_fraction=epoch_clip_frac / denom,
            grad_norm=epoch_grad_norm / denom,
            weight_norm=weight_norm,
            approx_kl=epoch_approx_kl / denom,
            entropy_coeff=ent_coeff,
            kl_early_stopped=kl_early_stopped,
        )

    # ------------------------------------------------------------------
    # High-level training loop with optional curriculum
    # ------------------------------------------------------------------

    def train(
        self,
        num_iterations: int,
        curriculum: Any | None = None,
        eval_interval: int = 0,
        log_fn: Any | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_metric: str = "mean_reward",
    ) -> list[dict[str, Any]]:
        """Run a complete training loop with optional curriculum scheduling.

        Parameters
        ----------
        num_iterations:
            Number of collect-rollout + update iterations.
        curriculum:
            Optional ``CurriculumScheduler`` — when provided the
            environment is rebuilt whenever the curriculum config changes.
        eval_interval:
            If > 0, run ``evaluate()`` every *eval_interval* iterations
            and include the result in the returned log.
        log_fn:
            Optional callback ``(iteration, log_entry) -> None`` called
            after each iteration for external logging.
        checkpoint_dir:
            If set, the best model (by *checkpoint_metric*) is saved to
            this directory automatically during training.
        checkpoint_metric:
            The log entry key used to decide the "best" model.
            Default ``"mean_reward"`` (higher is better).

        Returns
        -------
        list[dict[str, Any]]
            Per-iteration log entries with rewards, update metrics, LR,
            diagnostics, and optional eval metrics.
        """
        import os

        self.mappo_cfg.total_iterations = max(
            self.mappo_cfg.total_iterations, num_iterations
        )
        history: list[dict[str, Any]] = []
        prev_cfg_key: str | None = None
        best_metric: float = float("-inf")

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for it in range(num_iterations):
            # --- Curriculum: rebuild env if config changes ---
            if curriculum is not None:
                cur_cfg = curriculum.get_config(it, num_iterations)
                cfg_key = str(sorted(cur_cfg.items()))
                if cfg_key != prev_cfg_key:
                    self._rebuild_env(cur_cfg)
                    prev_cfg_key = cfg_key

            # --- Collect + Update ---
            rollout_info = self.collect_rollout()
            update_info = self.update()

            entry: dict[str, Any] = {
                "iteration": it,
                "mean_reward": rollout_info["mean_reward"],
                "total_reward": rollout_info["total_reward"],
                "lr": self.current_lr,
                "entropy_coeff": self.current_entropy_coeff,
            }
            for agent_type, res in update_info.items():
                entry[f"{agent_type}_policy_loss"] = res.policy_loss
                entry[f"{agent_type}_value_loss"] = res.value_loss
                entry[f"{agent_type}_entropy"] = res.entropy
                entry[f"{agent_type}_clip_frac"] = res.clip_fraction
                entry[f"{agent_type}_grad_norm"] = res.grad_norm
                entry[f"{agent_type}_approx_kl"] = res.approx_kl
                entry[f"{agent_type}_kl_early_stopped"] = res.kl_early_stopped

            # --- Optional eval ---
            if eval_interval > 0 and (it + 1) % eval_interval == 0:
                eval_metrics = self.evaluate()
                entry["eval"] = eval_metrics

            history.append(entry)
            if log_fn is not None:
                log_fn(it, entry)

            # --- Auto-checkpoint best model ---
            if checkpoint_dir is not None:
                metric_val = entry.get(checkpoint_metric, float("-inf"))
                if isinstance(metric_val, (int, float)) and metric_val > best_metric:
                    best_metric = metric_val
                    prefix = os.path.join(checkpoint_dir, "best")
                    self.save_models(prefix)

        return history

    def _rebuild_env(self, new_config: dict[str, Any]) -> None:
        """Rebuild the environment when curriculum config changes.

        Preserves the actor-critic networks and normaliser state but
        rebuilds the env + buffers for the new topology.  If observation
        dimensions change, networks are rebuilt from scratch.
        """
        from .config import get_default_config

        new_config = dict(new_config)
        new_config.setdefault("rollout_steps", self.mappo_cfg.rollout_length + 5)
        new_cfg = get_default_config(**new_config)
        new_obs_dims = obs_dim_from_env(new_cfg)
        new_global_dim = global_state_dim_from_config(new_cfg)

        dims_changed = (
            new_obs_dims != self.obs_dims
            or new_global_dim != self.global_dim
        )

        self.env = MaritimeEnv(config=new_cfg, seed=self.env.seed)
        self.cfg = self.env.cfg
        self.obs_dims = new_obs_dims
        self.global_dim = new_global_dim

        if dims_changed:
            # Must rebuild networks — old weights are incompatible
            self.actor_critics = build_actor_critics(
                config=self.cfg,
                vessel_obs_dim=self.obs_dims["vessel"],
                port_obs_dim=self.obs_dims["port"],
                coordinator_obs_dim=self.obs_dims["coordinator"],
                global_state_dim=self.global_dim,
                hidden_dims=self.mappo_cfg.hidden_dims,
            )
            for ac in self.actor_critics.values():
                ac.to(self.device)
            self.optimizers = {
                name: torch.optim.Adam(ac.parameters(), lr=self.mappo_cfg.lr)
                for name, ac in self.actor_critics.items()
            }
            # Reset normalizers for new dimensions
            self._obs_normalizers = {
                "vessel": ObsRunningMeanStd(self.obs_dims["vessel"]),
                "port": ObsRunningMeanStd(self.obs_dims["port"]),
                "coordinator": ObsRunningMeanStd(self.obs_dims["coordinator"]),
            }

        self._build_buffers()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        num_steps: int | None = None,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """Run a deterministic evaluation episode and return mean metrics."""
        num_steps = num_steps or int(self.cfg["rollout_steps"])
        obs = self.env.reset()

        vessel_ac = self.actor_critics["vessel"]
        port_ac = self.actor_critics["port"]
        coord_ac = self.actor_critics["coordinator"]
        vessel_ac.eval()
        port_ac.eval()
        coord_ac.eval()

        total_vessel_reward = 0.0
        total_port_reward = 0.0
        total_coord_reward = 0.0

        for _ in range(num_steps):
            global_state = self.env.get_global_state()
            gs_tensor = torch.as_tensor(
                global_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                # Vessels
                vessel_actions: list[dict[str, Any]] = []
                for v_obs in obs["vessels"]:
                    v_obs_n = self._eval_normalize_obs(v_obs, "vessel")
                    v_t = torch.as_tensor(
                        v_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    a, _, _ = vessel_ac.get_action_and_value(
                        v_t, gs_tensor, deterministic=deterministic
                    )
                    vessel_actions.append(_nn_to_vessel_action(a.squeeze(0), self.cfg))

                # Ports
                port_actions: list[dict[str, Any]] = []
                for i, p_obs in enumerate(obs["ports"]):
                    p_obs_n = self._eval_normalize_obs(p_obs, "port")
                    p_t = torch.as_tensor(
                        p_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    a, _, _ = port_ac.get_action_and_value(
                        p_t, gs_tensor, deterministic=deterministic
                    )
                    port_actions.append(_nn_to_port_action(a.squeeze(0), i, self.env))

                # Coordinators
                assignments = self.env._build_assignments()
                coord_actions: list[dict[str, Any]] = []
                for i, c_obs in enumerate(obs["coordinators"]):
                    c_obs_n = self._eval_normalize_obs(c_obs, "coordinator")
                    c_t = torch.as_tensor(
                        c_obs_n, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    a, _, _ = coord_ac.get_action_and_value(
                        c_t, gs_tensor, deterministic=deterministic
                    )
                    coord_actions.append(
                        _nn_to_coordinator_action(a.squeeze(0), i, self.env, assignments)
                    )

            actions_dict: dict[str, Any] = {
                "coordinator": coord_actions[0] if coord_actions else {},
                "coordinators": coord_actions,
                "vessels": vessel_actions,
                "ports": port_actions,
            }
            obs, rewards, done, _ = self.env.step(actions_dict)
            total_vessel_reward += float(np.mean(rewards["vessels"]))
            total_port_reward += float(np.mean(rewards["ports"]))
            total_coord_reward += float(rewards["coordinator"])
            if done:
                break

        return {
            "mean_vessel_reward": total_vessel_reward / num_steps,
            "mean_port_reward": total_port_reward / num_steps,
            "mean_coordinator_reward": total_coord_reward / num_steps,
            "total_reward": total_vessel_reward + total_port_reward + total_coord_reward,
        }

    def evaluate_episodes(
        self,
        num_episodes: int = 5,
        num_steps: int | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Run multiple evaluation episodes and return aggregated statistics.

        Returns per-metric mean, std, min, and max across episodes,
        plus the raw per-episode results.

        Parameters
        ----------
        num_episodes:
            Number of independent episodes to run.
        num_steps:
            Steps per episode (defaults to env rollout_steps).
        deterministic:
            Use deterministic (greedy) actions.

        Returns
        -------
        dict with keys:
            ``mean``, ``std``, ``min``, ``max`` — each a dict of floats.
            ``episodes`` — list of per-episode metric dicts.
        """
        episodes: list[dict[str, float]] = []
        for _ in range(num_episodes):
            ep_result = self.evaluate(
                num_steps=num_steps,
                deterministic=deterministic,
            )
            episodes.append(ep_result)

        if not episodes:
            return {"mean": {}, "std": {}, "min": {}, "max": {}, "episodes": []}

        keys = list(episodes[0].keys())
        mean_d: dict[str, float] = {}
        std_d: dict[str, float] = {}
        min_d: dict[str, float] = {}
        max_d: dict[str, float] = {}
        for k in keys:
            vals = np.array([ep[k] for ep in episodes])
            mean_d[k] = float(np.mean(vals))
            std_d[k] = float(np.std(vals))
            min_d[k] = float(np.min(vals))
            max_d[k] = float(np.max(vals))

        return {
            "mean": mean_d,
            "std": std_d,
            "min": min_d,
            "max": max_d,
            "episodes": episodes,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_buffer(self, agent_type: str) -> MultiAgentRolloutBuffer:
        if agent_type == "vessel":
            return self.vessel_buf
        elif agent_type == "port":
            return self.port_buf
        elif agent_type == "coordinator":
            return self.coordinator_buf
        raise ValueError(f"Unknown agent type: {agent_type}")

    def _reset_buffers(self) -> None:
        self.vessel_buf.reset()
        self.port_buf.reset()
        self.coordinator_buf.reset()

    def save_models(self, path_prefix: str) -> None:
        """Save actor-critic state dicts and normaliser state."""
        for name, ac in self.actor_critics.items():
            torch.save(ac.state_dict(), f"{path_prefix}_{name}.pt")
        # Persist normaliser state for reproducible evaluation
        import json
        norm_state: dict[str, Any] = {}
        for name, rms in self._reward_normalizers.items():
            norm_state[f"reward_{name}"] = {
                "mean": rms.mean, "var": rms.var, "count": rms.count,
            }
        for name, orms in self._obs_normalizers.items():
            norm_state[f"obs_{name}"] = {
                "mean": orms.mean.tolist(),
                "var": orms.var.tolist(),
                "count": orms.count,
            }
        with open(f"{path_prefix}_normalizers.json", "w") as f:
            json.dump(norm_state, f)

    def load_models(self, path_prefix: str) -> None:
        """Load actor-critic state dicts and normaliser state."""
        for name, ac in self.actor_critics.items():
            state_dict = torch.load(
                f"{path_prefix}_{name}.pt",
                map_location=self.device,
                weights_only=True,
            )
            ac.load_state_dict(state_dict)
        # Restore normaliser state if available
        import json
        import os
        norm_path = f"{path_prefix}_normalizers.json"
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                norm_state = json.load(f)
            for name, rms in self._reward_normalizers.items():
                key = f"reward_{name}"
                if key in norm_state:
                    rms.mean = float(norm_state[key]["mean"])
                    rms.var = float(norm_state[key]["var"])
                    rms.count = float(norm_state[key]["count"])
            for name, orms in self._obs_normalizers.items():
                key = f"obs_{name}"
                if key in norm_state:
                    orms.mean = np.array(norm_state[key]["mean"], dtype=np.float64)
                    orms.var = np.array(norm_state[key]["var"], dtype=np.float64)
                    orms.count = float(norm_state[key]["count"])

    @property
    def reward_history(self) -> list[float]:
        """Return episode mean rewards from all collected rollouts."""
        return list(self._episode_rewards)

    def get_diagnostics(self) -> dict[str, float]:
        """Return a snapshot of training diagnostics.

        Useful for logging between iterations.  Returns weight norms
        and reward-normaliser statistics for each agent type.
        """
        diag: dict[str, float] = {}
        for name, ac in self.actor_critics.items():
            total_norm = sum(p.data.norm().item() ** 2 for p in ac.parameters()) ** 0.5
            diag[f"{name}_weight_norm"] = total_norm
            total_params = sum(p.numel() for p in ac.parameters())
            diag[f"{name}_param_count"] = float(total_params)
        for name, rms in self._reward_normalizers.items():
            diag[f"{name}_reward_mean"] = rms.mean
            diag[f"{name}_reward_std"] = rms.std
        for name, orms in self._obs_normalizers.items():
            diag[f"{name}_obs_mean_norm"] = float(np.linalg.norm(orms.mean))
            diag[f"{name}_obs_std_mean"] = float(np.sqrt(np.maximum(orms.var, 0.0)).mean())
        diag["iteration"] = float(self._iteration)
        diag["lr"] = self.current_lr
        return diag
