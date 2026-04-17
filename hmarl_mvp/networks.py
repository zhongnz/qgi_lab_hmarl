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
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.mean_net = _make_mlp(obs_dim, hidden_dims, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return a Normal distribution over actions."""
        mean = self.mean_net(obs)
        std = self.log_std.clamp(-2.0, 0.5).exp().expand_as(mean)
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
            # Safety: if any row has all actions masked, allow all actions for
            # that row to avoid NaN from softmax([-inf, ...]).
            all_masked = ~action_mask.any(dim=-1, keepdim=True)
            safe_mask = action_mask | all_masked.expand_as(action_mask)
            logits = logits.masked_fill(~safe_mask, float("-inf"))
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
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return ``(action, log_prob, value, new_hidden)``.

        Parameters
        ----------
        action_mask:
            Optional boolean mask for discrete actors (True = valid).
            Ignored for continuous actors.
        hidden:
            GRU hidden state for ``RecurrentContinuousActor``.
            Shape ``(1, batch, hidden_size)``.  Ignored for other actors.
        """
        new_hidden: torch.Tensor | None = None
        if isinstance(self.actor, RecurrentContinuousActor):
            action, log_prob, new_hidden = self.actor.get_action(
                obs, deterministic=deterministic, hidden=hidden
            )
        elif isinstance(self.actor, (DiscreteActor, AttentionCoordinatorActor)):
            action, log_prob = self.actor.get_action(
                obs, deterministic=deterministic, action_mask=action_mask
            )
        else:
            action, log_prob = self.actor.get_action(obs, deterministic=deterministic)
        # Agent-conditioned critics receive both global state and local obs.
        if isinstance(self.critic, AgentConditionedCritic):
            value = self.critic(global_state, obs)
        else:
            value = self.critic(global_state)
        return action, log_prob, value, new_hidden

    def evaluate(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` for given transitions.

        Parameters
        ----------
        action_mask:
            Optional boolean mask for discrete actors (True = valid).
            Ignored for continuous actors.
        hidden:
            GRU hidden state for ``RecurrentContinuousActor``.
            Used during PPO updates to reconstruct the recurrent context.
        """
        if isinstance(self.actor, (DiscreteActor, AttentionCoordinatorActor)):
            log_prob, entropy = self.actor.evaluate(obs, actions, action_mask=action_mask)
        elif isinstance(self.actor, RecurrentContinuousActor):
            log_prob, entropy = self.actor.evaluate(obs, actions, hidden=hidden)
        else:
            log_prob, entropy = self.actor.evaluate(obs, actions)
        if isinstance(self.critic, AgentConditionedCritic):
            value = self.critic(global_state, obs)
        else:
            value = self.critic(global_state)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Medium-term architectural modules (§12.2)
# ---------------------------------------------------------------------------


class AttentionCoordinatorActor(nn.Module):
    """Attention-based coordinator actor with per-vessel action heads.

    Processes vessel and port features as entity sets using multi-head
    self-attention, then outputs independent per-vessel port assignments.

    The coordinator observation is split into:
    - Global features (5 dims: step + port-level aggregates)
    - Per-vessel features (7 dims each × num_vessels)
    - Per-port features (medium_horizon_days + 5 dims each × num_ports)
    - Optional weather matrix (num_ports × num_ports)

    After the transformer, each vessel token is fed through a shared
    action head that outputs logits over ``num_ports``.  Actions are
    sampled independently per vessel (factored action space).
    """

    def __init__(
        self,
        num_vessels: int,
        num_ports: int,
        vessel_entity_dim: int = 7,
        port_entity_dim: int = 10,
        global_feature_dim: int = 1,
        num_actions: int = 5,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        weather_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.num_vessels = num_vessels
        self.num_ports = num_ports
        self.vessel_entity_dim = vessel_entity_dim
        self.port_entity_dim = port_entity_dim
        self.global_feature_dim = global_feature_dim
        self.weather_enabled = weather_enabled
        self.weather_dim = num_ports * num_ports if weather_enabled else 0

        # Entity encoders project heterogeneous features to a common embedding space
        self.vessel_encoder = nn.Sequential(
            nn.Linear(vessel_entity_dim, embed_dim),
            nn.Tanh(),
        )
        self.port_encoder = nn.Sequential(
            nn.Linear(port_entity_dim, embed_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim + self.weather_dim, embed_dim),
            nn.Tanh(),
        )

        # Self-attention over all entity tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Per-vessel action head: each vessel token → logits over num_ports
        self.per_vessel_head = nn.Linear(embed_dim, num_ports)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split flat coordinator observation into entity groups."""
        idx = 0
        global_feat = obs[:, idx : idx + self.global_feature_dim]
        idx += self.global_feature_dim
        # Weather matrix (if present)
        weather_feat = obs[:, idx : idx + self.weather_dim] if self.weather_dim > 0 else None
        if self.weather_dim > 0:
            idx += self.weather_dim

        # Port features: num_ports * port_entity_dim
        port_total = self.num_ports * self.port_entity_dim
        port_flat = obs[:, idx : idx + port_total]
        idx += port_total

        # Vessel features: num_vessels * vessel_entity_dim
        vessel_total = self.num_vessels * self.vessel_entity_dim
        vessel_flat = obs[:, idx : idx + vessel_total]

        # Concatenate weather to global if present
        if weather_feat is not None:
            global_feat = torch.cat([global_feat, weather_feat], dim=-1)

        return (
            vessel_flat.view(-1, self.num_vessels, self.vessel_entity_dim),
            port_flat.view(-1, self.num_ports, self.port_entity_dim),
            global_feat,
        )

    def _get_per_vessel_logits(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-vessel logits of shape ``(B, V, num_ports)``."""
        vessel_ents, port_ents, global_feat = self._split_obs(obs)

        vessel_tokens = self.vessel_encoder(vessel_ents)   # (B, V, E)
        port_tokens = self.port_encoder(port_ents)         # (B, P, E)
        global_token = self.global_encoder(global_feat).unsqueeze(1)  # (B, 1, E)

        tokens = torch.cat([global_token, vessel_tokens, port_tokens], dim=1)
        attended = self.transformer(tokens)

        # Extract vessel tokens (indices 1..V after the global token)
        vessel_attended = attended[:, 1 : 1 + self.num_vessels, :]  # (B, V, E)
        logits = self.per_vessel_head(vessel_attended)  # (B, V, num_ports)

        if action_mask is not None:
            # action_mask: (B, V, P) per-vessel or (B, P) shared
            if action_mask.ndim == 2:
                mask = action_mask.unsqueeze(1).expand_as(logits)
            else:
                mask = action_mask.expand_as(logits)
            all_masked = ~mask.any(dim=-1, keepdim=True)
            safe_mask = mask | all_masked.expand_as(mask)
            logits = logits.masked_fill(~safe_mask, float("-inf"))

        return logits

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """Return per-vessel Categorical distributions.

        The returned ``Categorical`` has ``logits`` of shape
        ``(B, V, num_ports)`` so that ``dist.sample()`` yields
        ``(B, V)`` and ``dist.log_prob(actions)`` expects ``(B, V)``.
        """
        logits = self._get_per_vessel_logits(obs, action_mask)
        return torch.distributions.Categorical(logits=logits)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-vessel actions and return joint log-prob.

        Returns:
            actions: ``(B, V)`` — per-vessel port indices.
            log_prob: ``(B,)`` — sum of per-vessel log probs.
        """
        dist = self.forward(obs, action_mask=action_mask)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)  # (B, V)
        else:
            actions = dist.sample()  # (B, V)
        # Joint log-prob = sum of independent per-vessel log-probs
        per_vessel_lp = dist.log_prob(actions)  # (B, V)
        log_prob = per_vessel_lp.sum(dim=-1)  # (B,)
        return actions, log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate per-vessel actions, returning joint log-prob and entropy.

        Args:
            actions: ``(B, V)`` — per-vessel port indices.

        Returns:
            log_prob: ``(B,)`` — sum of per-vessel log probs.
            entropy: ``(B,)`` — sum of per-vessel entropies.
        """
        dist = self.forward(obs, action_mask=action_mask)
        acts = actions.long()
        if acts.dim() == 1:
            acts = acts.unsqueeze(0)
        if acts.dim() == 3:
            acts = acts.squeeze(-1)
        per_vessel_lp = dist.log_prob(acts)  # (B, V)
        per_vessel_ent = dist.entropy()  # (B, V)
        return per_vessel_lp.sum(dim=-1), per_vessel_ent.sum(dim=-1)


class EncodedCritic(nn.Module):
    """Critic with separate per-type observation encoders.

    Instead of feeding the raw 464-dim global state through a single MLP,
    this critic splits the global state into per-type segments, encodes
    each through a type-specific encoder, then concatenates and feeds
    through a shared value head. This reduces representational burden
    and allows each encoder to specialise.

    Global state layout (with weather_enabled=True):
    - coordinator obs (132 dims)
    - vessel obs × num_vessels (27 dims each)
    - port obs × num_ports (22 dims each)
    - time features (5 dims)
    - weather flag (1 dim)
    """

    def __init__(
        self,
        coordinator_obs_dim: int,
        vessel_obs_dim: int,
        port_obs_dim: int,
        num_vessels: int,
        num_ports: int,
        extra_dims: int = 6,
        encoder_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.coordinator_obs_dim = coordinator_obs_dim
        self.vessel_obs_dim = vessel_obs_dim
        self.port_obs_dim = port_obs_dim
        self.num_vessels = num_vessels
        self.num_ports = num_ports
        self.extra_dims = extra_dims

        hidden_dims = hidden_dims or [64, 64]

        self.coord_encoder = nn.Sequential(
            nn.Linear(coordinator_obs_dim, encoder_dim),
            nn.Tanh(),
        )
        self.vessel_encoder = nn.Sequential(
            nn.Linear(vessel_obs_dim, encoder_dim),
            nn.Tanh(),
        )
        self.port_encoder = nn.Sequential(
            nn.Linear(port_obs_dim, encoder_dim),
            nn.Tanh(),
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(extra_dims, encoder_dim),
            nn.Tanh(),
        )

        # Shared value head over concatenated encoded representations
        # 4 encoder outputs → 4 * encoder_dim input
        combined_dim = 4 * encoder_dim
        self.value_head = _make_mlp(combined_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s) from the encoded global state."""
        idx = 0
        coord_obs = state[:, idx : idx + self.coordinator_obs_dim]
        idx += self.coordinator_obs_dim

        # Vessel observations: encode each then mean-pool
        vessel_total = self.num_vessels * self.vessel_obs_dim
        vessel_flat = state[:, idx : idx + vessel_total]
        idx += vessel_total
        vessel_reshaped = vessel_flat.view(-1, self.num_vessels, self.vessel_obs_dim)
        vessel_encoded = self.vessel_encoder(vessel_reshaped).mean(dim=1)

        # Port observations: encode each then mean-pool
        port_total = self.num_ports * self.port_obs_dim
        port_flat = state[:, idx : idx + port_total]
        idx += port_total
        port_reshaped = port_flat.view(-1, self.num_ports, self.port_obs_dim)
        port_encoded = self.port_encoder(port_reshaped).mean(dim=1)

        # Extra global features (time features + weather flag)
        extra = state[:, idx : idx + self.extra_dims]

        coord_enc = self.coord_encoder(coord_obs)
        extra_enc = self.extra_encoder(extra)

        combined = torch.cat([coord_enc, vessel_encoded, port_encoded, extra_enc], dim=-1)
        return self.value_head(combined).squeeze(-1)


class AgentConditionedCritic(nn.Module):
    """EncodedCritic wrapper that also conditions on the individual agent's obs.

    Standard MAPPO centralized critics receive the global state but cannot
    distinguish which agent's value they are predicting.  For homogeneous
    agents with parameter sharing this is a known limitation — the critic
    averages over the fleet.

    This wrapper adds a local-observation encoder whose output is
    concatenated with the EncodedCritic's combined representation before
    the value head.  This lets the vessel critic learn "given the global
    fleet state AND my own position/speed/fuel, what is MY value?"

    The local obs is passed via a second argument to ``forward(state, obs)``.
    ``ActorCritic`` is modified to pass both when this critic type is used.
    """

    def __init__(
        self,
        base_critic: EncodedCritic,
        agent_obs_dim: int,
        encoder_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.base = base_critic
        hidden_dims = hidden_dims or [64, 64]

        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_dim, encoder_dim),
            nn.Tanh(),
        )

        # Replace the base critic's value head with one that also
        # takes the agent encoding as input.
        base_combined_dim = 4 * encoder_dim  # from EncodedCritic
        self.value_head = _make_mlp(
            base_combined_dim + encoder_dim,  # +agent_enc
            hidden_dims,
            1,
        )

    def forward(self, state: torch.Tensor, agent_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Return V(s, o_i) conditioned on global state and agent's local obs."""
        # Run the base critic's encoders (but not its value head)
        idx = 0
        b = self.base
        coord_obs = state[:, idx : idx + b.coordinator_obs_dim]
        idx += b.coordinator_obs_dim

        vessel_total = b.num_vessels * b.vessel_obs_dim
        vessel_flat = state[:, idx : idx + vessel_total]
        idx += vessel_total
        vessel_reshaped = vessel_flat.view(-1, b.num_vessels, b.vessel_obs_dim)
        vessel_encoded = b.vessel_encoder(vessel_reshaped).mean(dim=1)

        port_total = b.num_ports * b.port_obs_dim
        port_flat = state[:, idx : idx + port_total]
        idx += port_total
        port_reshaped = port_flat.view(-1, b.num_ports, b.port_obs_dim)
        port_encoded = b.port_encoder(port_reshaped).mean(dim=1)

        extra = state[:, idx : idx + b.extra_dims]
        coord_enc = b.coord_encoder(coord_obs)
        extra_enc = b.extra_encoder(extra)

        combined = torch.cat([coord_enc, vessel_encoded, port_encoded, extra_enc], dim=-1)

        if agent_obs is not None:
            agent_enc = self.agent_encoder(agent_obs)
            combined = torch.cat([combined, agent_enc], dim=-1)
            return self.value_head(combined).squeeze(-1)
        else:
            # Fallback: use base critic's value head (no agent conditioning)
            return b.value_head(combined).squeeze(-1)


class RecurrentContinuousActor(nn.Module):
    """GRU-based continuous actor for temporal reasoning.

    Replaces the feed-forward MLP actor for vessels, adding a GRU
    cell that maintains a hidden state across steps within a rollout.
    This enables the vessel to reason about temporal patterns like
    instruction staleness and congestion trends.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mean_head = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """Return (distribution, new_hidden_state).

        Parameters
        ----------
        obs:
            Shape ``(batch, obs_dim)`` or ``(batch, seq, obs_dim)``.
        hidden:
            GRU hidden state ``(1, batch, hidden_size)`` or None.
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
        x = self.input_proj(obs)     # (B, S, H)
        out, new_hidden = self.gru(x, hidden)  # out: (B, S, H)
        # Use last timestep for action
        last = out[:, -1, :]         # (B, H)
        mean = self.mean_head(last)
        std = self.log_std.clamp(-2.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), new_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, new_hidden)``."""
        dist, new_hidden = self.forward(obs, hidden)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, new_hidden

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy (hidden state not returned)."""
        dist, _ = self.forward(obs, hidden)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Return zero-initialised hidden state."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


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
    vessel_hidden_dims: list[int] | None = None,
    port_hidden_dims: list[int] | None = None,
    coordinator_hidden_dims: list[int] | None = None,
    coordinator_use_attention: bool = False,
    use_encoded_critic: bool = False,
    vessel_use_recurrence: bool = False,
) -> dict[str, ActorCritic]:
    """Create one ``ActorCritic`` per agent type for the HMARL hierarchy.

    Action spaces (based on current policy interface):
    - Vessel: continuous ``[target_speed, requested_arrival_time]`` (2-D)
    - Port: discrete joint ``(service_rate, accept_requests)``
      with ``(docks + 1)^2`` choices
    - Coordinator: discrete ``dest_port × departure_window`` bins.

    Architectural flags:
    - ``coordinator_use_attention``: swap coordinator actor with
      ``AttentionCoordinatorActor``.
    - ``use_encoded_critic``: swap all critics with ``EncodedCritic``.
    - ``vessel_use_recurrence``: swap vessel actor with
      ``RecurrentContinuousActor``.
    """
    hidden_dims = hidden_dims or [64, 64]
    v_hd = vessel_hidden_dims or hidden_dims
    p_hd = port_hidden_dims or hidden_dims
    c_hd = coordinator_hidden_dims or hidden_dims
    dep_windows = config.get("coordinator_departure_window_options", (0,))
    num_windows = len(dep_windows) if isinstance(dep_windows, (list, tuple)) else 1
    port_action_dim = (int(config["docks_per_port"]) + 1) ** 2
    coordinator_actions = int(config["num_ports"]) * max(int(num_windows), 1)
    num_vessels = int(config["num_vessels"])
    num_ports = int(config["num_ports"])

    result: dict[str, ActorCritic] = {
        "vessel": ActorCritic(
            obs_dim=vessel_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=2,
            discrete=False,
            hidden_dims=v_hd,
        ),
        "port": ActorCritic(
            obs_dim=port_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=port_action_dim,
            discrete=True,
            hidden_dims=p_hd,
        ),
        "coordinator": ActorCritic(
            obs_dim=coordinator_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=coordinator_actions,
            discrete=True,
            hidden_dims=c_hd,
        ),
    }

    # --- Architectural swaps ---

    if vessel_use_recurrence:
        result["vessel"].actor = RecurrentContinuousActor(
            obs_dim=vessel_obs_dim,
            act_dim=2,
            hidden_size=v_hd[0] if v_hd else 64,
        )

    if coordinator_use_attention:
        weather_enabled = config.get("weather_enabled", True)
        medium_horizon = config.get("medium_horizon_days", 5)
        port_entity_dim = medium_horizon + 5
        result["coordinator"].actor = AttentionCoordinatorActor(
            num_vessels=num_vessels,
            num_ports=num_ports,
            vessel_entity_dim=7,
            port_entity_dim=port_entity_dim,
            global_feature_dim=1,
            embed_dim=c_hd[0] if c_hd else 64,
            weather_enabled=weather_enabled,
        )

    if use_encoded_critic:
        # Extra dims at end of global state: global_congestion(num_ports) + total_emissions(1)
        extra_dims = num_ports + 1
        # Create separate EncodedCritic instances per agent type so that
        # vessel/port/coordinator value losses don't compete for the same
        # parameters.  Shared critics caused optimization conflicts that
        # degraded vessel EV (all three types pulling in different directions).
        #
        # The coordinator gets a much smaller critic (encoder_dim=16,
        # hidden=[32]) because it only generates ~10 transitions per
        # rollout (12-step cadence).  A large critic (~32K params) cannot
        # fit from 10 samples; a small one (~5K params) can.
        for agent_type, ac in result.items():
            hd = {"vessel": v_hd, "port": p_hd, "coordinator": c_hd}.get(agent_type, hidden_dims)
            if agent_type == "coordinator":
                # Small critic for data-starved coordinator
                coord_enc_dim = 16
                coord_critic_hd = [32]
                base_enc = EncodedCritic(
                    coordinator_obs_dim=coordinator_obs_dim,
                    vessel_obs_dim=vessel_obs_dim,
                    port_obs_dim=port_obs_dim,
                    num_vessels=num_vessels,
                    num_ports=num_ports,
                    extra_dims=extra_dims,
                    encoder_dim=coord_enc_dim,
                    hidden_dims=coord_critic_hd,
                )
                ac.critic = base_enc
            else:
                enc_dim = hd[0] if hd else 64
                base_enc = EncodedCritic(
                    coordinator_obs_dim=coordinator_obs_dim,
                    vessel_obs_dim=vessel_obs_dim,
                    port_obs_dim=port_obs_dim,
                    num_vessels=num_vessels,
                    num_ports=num_ports,
                    extra_dims=extra_dims,
                    encoder_dim=enc_dim,
                    hidden_dims=hd,
                )
                if agent_type == "vessel":
                    # Wrap vessel critic with agent conditioning so it can
                    # distinguish individual vessel states (position, speed,
                    # fuel) from the fleet average.
                    ac.critic = AgentConditionedCritic(
                        base_critic=base_enc,
                        agent_obs_dim=vessel_obs_dim,
                        encoder_dim=enc_dim,
                        hidden_dims=hd,
                    )
                else:
                    ac.critic = base_enc

    return result


def build_per_agent_actor_critics(
    config: dict[str, Any],
    vessel_obs_dim: int,
    port_obs_dim: int,
    coordinator_obs_dim: int,
    global_state_dim: int,
    hidden_dims: list[int] | None = None,
    vessel_hidden_dims: list[int] | None = None,
    port_hidden_dims: list[int] | None = None,
    coordinator_hidden_dims: list[int] | None = None,
) -> dict[str, ActorCritic]:
    """Create separate ``ActorCritic`` networks for each individual agent.

    Unlike ``build_actor_critics`` which creates one network per agent *type*
    (shared across all agents of that type), this creates one network per
    individual agent: ``vessel_0``, ``vessel_1``, ..., ``port_0``, etc.

    This is the *no parameter sharing* ablation — useful for comparing
    against the default shared-parameter CTDE architecture.
    """
    hidden_dims = hidden_dims or [64, 64]
    v_hd = vessel_hidden_dims or hidden_dims
    p_hd = port_hidden_dims or hidden_dims
    c_hd = coordinator_hidden_dims or hidden_dims
    dep_windows = config.get("coordinator_departure_window_options", (0,))
    num_windows = len(dep_windows) if isinstance(dep_windows, (list, tuple)) else 1
    port_action_dim = (int(config["docks_per_port"]) + 1) ** 2
    coordinator_actions = int(config["num_ports"]) * max(int(num_windows), 1)
    nets: dict[str, ActorCritic] = {}

    for i in range(config["num_vessels"]):
        nets[f"vessel_{i}"] = ActorCritic(
            obs_dim=vessel_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=2,
            discrete=False,
            hidden_dims=v_hd,
        )

    for i in range(config["num_ports"]):
        nets[f"port_{i}"] = ActorCritic(
            obs_dim=port_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=port_action_dim,
            discrete=True,
            hidden_dims=p_hd,
        )

    for i in range(config.get("num_coordinators", 1)):
        nets[f"coordinator_{i}"] = ActorCritic(
            obs_dim=coordinator_obs_dim,
            global_state_dim=global_state_dim,
            act_dim=coordinator_actions,
            discrete=True,
            hidden_dims=c_hd,
        )

    return nets


def obs_dim_from_env(
    config: dict[str, Any],
) -> dict[str, int]:
    """Compute observation dimensions for each agent type.

    Based on the observation builders in ``agents.py`` and ``env.py``:
    - Vessel: 12 (local: vessel_id_norm, loc, position_nm, speed, fuel,
              emissions, stalled, port_service_state, dock_avail, at_sea,
              remaining_range_nm, deadline_delta_hours)
              + short_horizon_hours (forecast) + 3 (directive)
              + 1 if weather_enabled (sea_state)
    - Port: 6 (local: queue, docks, occupied, booked_arrivals,
              imminent_arrivals, occupancy_rate)
              + short_horizon_hours (forecast) + 1 (incoming)
            + 3 if weather_enabled and port_weather_features
    - Coordinator: num_ports * medium_horizon_days + num_ports * 5
                   + num_vessels * 7 + 1
                   + num_ports * num_ports if weather_enabled
                   (flattened weather matrix)
    """
    short_h = config["short_horizon_hours"]
    num_ports = config["num_ports"]
    medium_d = config["medium_horizon_days"]
    num_vessels = config["num_vessels"]

    vessel_dim = 12 + short_h + 3  # 12 = 11 local features + 1 vessel_id
    if config.get("weather_enabled", True):
        vessel_dim += 1  # sea_state on current route
    port_dim = 6 + short_h + 1
    if config.get("weather_enabled", True) and config.get("port_weather_features", True):
        port_dim += 3  # inbound weather summary (mean, max, rough_fraction)
    coordinator_dim = num_ports * medium_d + num_ports * 5 + num_vessels * 7 + 1
    if config.get("weather_enabled", True):
        coordinator_dim += num_ports * num_ports

    return {
        "vessel": vessel_dim,
        "port": port_dim,
        "coordinator": coordinator_dim,
    }
