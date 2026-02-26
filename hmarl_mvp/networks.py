"""Actor-critic neural network modules for MAPPO with CTDE."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Orthogonal initialisation (PPO best-practice)
# ---------------------------------------------------------------------------


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Apply orthogonal initialisation to a single ``nn.Linear`` layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_orthogonal_init(
    model: nn.Module,
    gain: float = math.sqrt(2),
    output_gain: float = 0.01,
) -> None:
    """Apply orthogonal initialisation to all layers in *model*.

    Hidden layers use ``gain`` (default √2 for ReLU/Tanh); the final
    output layer uses ``output_gain`` (small value for policy heads,
    1.0 for value heads).  The last ``nn.Linear`` found is treated as
    the output layer.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for layer in linears[:-1]:
        _ortho_init(layer, gain=gain)
    if linears:
        _ortho_init(linears[-1], gain=output_gain)


def _make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Build a simple feed-forward MLP."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ContinuousActor(nn.Module):
    """Gaussian actor for continuous action spaces (e.g. speed control).

    Outputs a mean vector; log-std is a learnable parameter vector.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] | None = None,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.mean_net = _make_mlp(obs_dim, hidden_dims, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return a Normal distribution over actions."""
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return ``(action, log_prob)``."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy for given obs-action pairs."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class DiscreteActor(nn.Module):
    """Categorical actor for discrete action spaces (e.g. port selection)."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.logit_net = _make_mlp(obs_dim, hidden_dims, num_actions)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """Return a Categorical distribution over actions.

        Parameters
        ----------
        obs:
            Observation tensor.
        action_mask:
            Optional boolean tensor (True = valid).  Invalid actions
            receive ``-inf`` logits so they are never sampled.
        """
        logits = self.logit_net(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return ``(action, log_prob)``."""
        dist = self.forward(obs, action_mask=action_mask)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy for given obs-action pairs."""
        dist = self.forward(obs, action_mask=action_mask)
        log_prob = dist.log_prob(actions.long().squeeze(-1))
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """State-value function V(s).

    For CTDE, the critic receives the global state (all agent observations +
    global info) rather than the local observation.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.value_net = _make_mlp(state_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s) as a 1-D tensor."""
        return self.value_net(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor + critic module for a single agent type.

    Parameters
    ----------
    obs_dim:
        Dimension of the agent's local observation (actor input).
    global_state_dim:
        Dimension of the global state (critic input for CTDE).
    act_dim:
        Dimension of the action space.
    discrete:
        If True, use ``DiscreteActor``; otherwise ``ContinuousActor``.
    hidden_dims:
        Hidden layer sizes for both actor and critic.
    """

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        act_dim: int,
        discrete: bool = False,
        hidden_dims: list[int] | None = None,
        ortho_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        if discrete:
            self.actor: DiscreteActor | ContinuousActor = DiscreteActor(
                obs_dim, act_dim, hidden_dims
            )
        else:
            self.actor = ContinuousActor(obs_dim, act_dim, hidden_dims)
        self.critic = Critic(global_state_dim, hidden_dims)

        if ortho_init:
            # Actor: small output gain for stable initial policy
            apply_orthogonal_init(self.actor, output_gain=0.01)
            # Critic: gain=1.0 output for unbiased initial value estimates
            apply_orthogonal_init(self.critic, output_gain=1.0)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, value)``.

        Parameters
        ----------
        action_mask:
            Optional boolean mask for discrete actors (True = valid).
            Ignored for continuous actors.
        """
        if isinstance(self.actor, DiscreteActor):
            action, log_prob = self.actor.get_action(
                obs, deterministic=deterministic, action_mask=action_mask
            )
        else:
            action, log_prob = self.actor.get_action(obs, deterministic=deterministic)
        value = self.critic(global_state)
        return action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` for given transitions.

        Parameters
        ----------
        action_mask:
            Optional boolean mask for discrete actors (True = valid).
            Ignored for continuous actors.
        """
        if isinstance(self.actor, DiscreteActor):
            log_prob, entropy = self.actor.evaluate(obs, actions, action_mask=action_mask)
        else:
            log_prob, entropy = self.actor.evaluate(obs, actions)
        value = self.critic(global_state)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_actor_critics(
    config: dict[str, Any],
    vessel_obs_dim: int,
    port_obs_dim: int,
    coordinator_obs_dim: int,
    global_state_dim: int,
    hidden_dims: list[int] | None = None,
) -> dict[str, ActorCritic]:
    """Create one ``ActorCritic`` per agent type for the HMARL hierarchy.

    Action spaces (based on current policy interface):
    - Vessel: continuous ``[target_speed]`` (1-D)
    - Port: discrete ``service_rate`` (docks + 1 choices)
    - Coordinator: discrete ``dest_port`` (num_ports choices)
    """
    hidden_dims = hidden_dims or [64, 64]
    return {
        "vessel": ActorCritic(
            obs_dim=vessel_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=1,
            discrete=False,
            hidden_dims=hidden_dims,
        ),
        "port": ActorCritic(
            obs_dim=port_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=config["docks_per_port"] + 1,
            discrete=True,
            hidden_dims=hidden_dims,
        ),
        "coordinator": ActorCritic(
            obs_dim=coordinator_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=config["num_ports"],
            discrete=True,
            hidden_dims=hidden_dims,
        ),
    }


def build_per_agent_actor_critics(
    config: dict[str, Any],
    vessel_obs_dim: int,
    port_obs_dim: int,
    coordinator_obs_dim: int,
    global_state_dim: int,
    hidden_dims: list[int] | None = None,
) -> dict[str, ActorCritic]:
    """Create separate ``ActorCritic`` networks for each individual agent.

    Unlike ``build_actor_critics`` which creates one network per agent *type*
    (shared across all agents of that type), this creates one network per
    individual agent: ``vessel_0``, ``vessel_1``, ..., ``port_0``, etc.

    This is the *no parameter sharing* ablation — useful for comparing
    against the default shared-parameter CTDE architecture.
    """
    hidden_dims = hidden_dims or [64, 64]
    nets: dict[str, ActorCritic] = {}

    for i in range(config["num_vessels"]):
        nets[f"vessel_{i}"] = ActorCritic(
            obs_dim=vessel_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=1,
            discrete=False,
            hidden_dims=hidden_dims,
        )

    for i in range(config["num_ports"]):
        nets[f"port_{i}"] = ActorCritic(
            obs_dim=port_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=config["docks_per_port"] + 1,
            discrete=True,
            hidden_dims=hidden_dims,
        )

    for i in range(config.get("num_coordinators", 1)):
        nets[f"coordinator_{i}"] = ActorCritic(
            obs_dim=coordinator_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=config["num_ports"],
            discrete=True,
            hidden_dims=hidden_dims,
        )

    return nets


def obs_dim_from_env(
    config: dict[str, Any],
) -> dict[str, int]:
    """Compute observation dimensions for each agent type.

    Based on the observation builders in ``agents.py`` and ``env.py``:
    - Vessel: 5 (local: loc, speed, fuel, emissions, dock_avail)
              + short_horizon_hours (forecast) + 3 (directive)
              + 1 if weather_enabled (sea_state)
    - Port: 3 (local) + short_horizon_hours (forecast) + 1 (incoming)
    - Coordinator: num_ports * medium_horizon_days + num_vessels * 4 + 1
    """
    short_h = config["short_horizon_hours"]
    num_ports = config["num_ports"]
    medium_d = config["medium_horizon_days"]
    num_vessels = config["num_vessels"]

    vessel_dim = 5 + short_h + 3  # 5 local (loc, speed, fuel, emissions, dock_avail)
    if config.get("weather_enabled", False):
        vessel_dim += 1  # sea_state on current route
    port_dim = 3 + short_h + 1
    coordinator_dim = num_ports * medium_d + num_vessels * 4 + 1

    return {
        "vessel": vessel_dim,
        "port": port_dim,
        "coordinator": coordinator_dim,
    }
