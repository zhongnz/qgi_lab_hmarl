"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent, assign_vessels_to_coordinators
from .config import SEED, DecisionCadence, get_default_config, resolve_distance_matrix
from .dynamics import dispatch_vessel, step_ports, step_vessels
from .forecasts import MediumTermForecaster, ShortTermForecaster
from .message_bus import MessageBus
from .metrics import compute_port_metrics
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_vessel_reward_step,
)
from .state import PortState, VesselState, initialize_ports, initialize_vessels, make_rng


class MaritimeEnv:
    """
    Gymnasium-style multi-agent maritime environment skeleton.

    The class intentionally uses plain dictionaries and numpy arrays to keep
    integration easy for both notebooks and future gymnasium wrappers.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int = SEED,
        distance_nm: np.ndarray | None = None,
    ) -> None:
        user_cfg = dict(config or {})
        if "seed" not in user_cfg:
            user_cfg["seed"] = seed
        self.cfg = get_default_config(**user_cfg)
        self.seed = int(self.cfg["seed"])
        self.rng = make_rng(self.seed)
        self.num_ports = self.cfg["num_ports"]
        self.num_vessels = self.cfg["num_vessels"]
        self.num_coordinators = max(1, int(self.cfg.get("num_coordinators", 1)))
        self.distance_nm = resolve_distance_matrix(self.num_ports, distance_nm)
        self.cadence = DecisionCadence.from_config(self.cfg)
        self.t = 0

        self.ports: list[PortState] = []
        self.vessels: list[VesselState] = []
        self.coordinators: list[FleetCoordinatorAgent] = []
        self.port_agents: list[PortAgent] = []
        self.vessel_agents: list[VesselAgent] = []
        self.medium_forecast: np.ndarray | None = None
        self.short_forecast: np.ndarray | None = None
        self.medium_forecaster = MediumTermForecaster(self.cfg["medium_horizon_days"])
        self.short_forecaster = ShortTermForecaster(self.cfg["short_horizon_hours"])

        self.bus = MessageBus(self.num_ports)
        self._last_port_actions: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        """Reset state and return initial observations."""
        self.t = 0
        self.rng = make_rng(self.seed)
        self.ports = initialize_ports(
            num_ports=self.num_ports,
            docks_per_port=self.cfg["docks_per_port"],
            rng=self.rng,
            service_time_hours=self.cfg["service_time_hours"],
        )
        self.vessels = initialize_vessels(
            num_vessels=self.num_vessels,
            num_ports=self.num_ports,
            nominal_speed=self.cfg["nominal_speed"],
            rng=self.rng,
            initial_fuel=self.cfg["initial_fuel"],
        )
        self.coordinators = [
            FleetCoordinatorAgent(config=self.cfg, coordinator_id=i)
            for i in range(self.num_coordinators)
        ]
        self.vessel_agents = [VesselAgent(v, self.cfg) for v in self.vessels]
        self.port_agents = [PortAgent(p, self.cfg) for p in self.ports]
        self._reset_runtime_state()
        self._refresh_forecasts()
        return self._get_observations()

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool, dict[str, Any]]:
        """Execute one environment tick."""
        due = self.cadence.due(self.t)
        delivered_responses = self.bus.deliver_due(self.t)
        assignments = self._build_assignments()
        step_delay_by_vessel: dict[int, float] = {
            vessel.vessel_id: 0.0 for vessel in self.vessels
        }
        step_requests_submitted = 0
        step_requests_accepted = 0
        step_requests_rejected = 0

        coordinator_actions = self._normalize_coordinator_actions(actions)
        if due["coordinator"]:
            for coordinator_id, coordinator in enumerate(self.coordinators):
                raw_action = coordinator_actions[coordinator_id]
                applied = coordinator.apply_action(raw_action)
                per_vessel_dest = raw_action.get("per_vessel_dest", {})
                enriched = {
                    **applied,
                    "coordinator_id": coordinator_id,
                    "assigned_vessel_ids": assignments.get(coordinator_id, []),
                }
                for vessel_id in assignments.get(coordinator_id, []):
                    vessel_directive = dict(enriched)
                    if vessel_id in per_vessel_dest:
                        vessel_directive["dest_port"] = int(per_vessel_dest[vessel_id])
                    self.bus.enqueue_directive(
                        self.t + self.cadence.message_latency_steps,
                        vessel_id,
                        vessel_directive,
                    )

        vessel_inputs = self._normalize_vessel_actions(actions)
        normalized_vessel_actions: list[dict[str, Any]] = []
        for idx, vessel_agent in enumerate(self.vessel_agents):
            vessel = vessel_agent.state
            vessel_id = vessel.vessel_id
            directive = self.bus.get_latest_directive(vessel_id)
            if directive is None:
                directive = self._fallback_directive_for_vessel(vessel_id, assignments)

            if due["vessel"]:
                normalized = vessel_agent.apply_action(vessel_inputs[idx])
                if (
                    normalized.get("request_arrival_slot", False)
                    and not vessel.at_sea
                    and not self.bus.is_awaiting(vessel_id)
                ):
                    destination = int(directive.get("dest_port", vessel.destination))
                    self.bus.enqueue_arrival_request(
                        self.t + self.cadence.message_latency_steps, vessel_id, destination
                    )
                    self.bus.mark_awaiting(vessel_id)
                    step_requests_submitted += 1
            else:
                normalized = dict(vessel_agent.last_action)

            response = delivered_responses.get(vessel_id)
            if response is not None:
                if response["accepted"] and not vessel.at_sea:
                    dispatch_vessel(
                        vessel=vessel,
                        destination=int(response["dest_port"]),
                        speed=float(normalized["target_speed"]),
                        config=self.cfg,
                    )
                elif not response["accepted"] and not vessel.at_sea:
                    vessel.delay_hours += 1.0
                    step_delay_by_vessel[vessel_id] += 1.0
                self.bus.clear_awaiting(vessel_id)
            elif self.bus.is_awaiting(vessel_id) and not vessel.at_sea:
                vessel.delay_hours += 1.0
                step_delay_by_vessel[vessel_id] += 1.0
            normalized_vessel_actions.append(normalized)

        pre_step_at_sea = {v.vessel_id: v.at_sea for v in self.vessels}

        vessel_step_stats = step_vessels(
            vessels=self.vessels,
            distance_nm=self.distance_nm,
            config=self.cfg,
            dt_hours=1.0,
        )

        for vessel in self.vessels:
            if (
                pre_step_at_sea.get(vessel.vessel_id, False)
                and not vessel.at_sea
                and vessel.location == vessel.destination
            ):
                self.ports[vessel.location].queue += 1

        port_inputs = self._normalize_port_actions(actions)
        if due["port"]:
            normalized_port_actions: list[dict[str, Any]] = []
            for port_id, (port_agent, action) in enumerate(zip(self.port_agents, port_inputs)):
                normalized = port_agent.apply_action(action)
                normalized_port_actions.append(normalized)
                backlog = self.bus.get_pending_requests(port_id)
                available_slots = max(port_agent.state.docks - port_agent.state.occupied, 0)
                accept_limit = min(
                    len(backlog),
                    max(int(normalized.get("accept_requests", 0)), 0),
                    available_slots,
                )
                accepted = backlog[:accept_limit]
                rejected = backlog[accept_limit:]
                step_requests_accepted += len(accepted)
                step_requests_rejected += len(rejected)
                self.bus.clear_pending_requests(port_id)
                response_step = self.t + self.cadence.message_latency_steps
                for vessel_id in accepted:
                    self.bus.enqueue_slot_response(response_step, vessel_id, True, port_id)
                for vessel_id in rejected:
                    self.bus.enqueue_slot_response(response_step, vessel_id, False, port_id)
            self._last_port_actions = normalized_port_actions

        service_rates = [
            int(action.get("service_rate", 1))
            for action in (
                self._last_port_actions
                if self._last_port_actions
                else [port_agent.last_action for port_agent in self.port_agents]
            )
        ]
        step_ports(
            self.ports,
            service_rates,
            dt_hours=1.0,
            service_time_hours=self.cfg["service_time_hours"],
        )

        rewards = self._compute_rewards(
            vessel_step_stats=vessel_step_stats,
            step_delay_by_vessel=step_delay_by_vessel,
        )

        self.t += 1
        done = self.t >= self.cfg["rollout_steps"]
        self._refresh_forecasts()
        obs = self._get_observations()
        info = {
            "port_metrics": compute_port_metrics(self.ports),
            "cadence_due": due,
            "coordinator_assignments": assignments,
            "message_queues": self.bus.queue_sizes,
            "pending_arrival_requests": float(self.bus.total_pending_requests),
            "requests_submitted": float(step_requests_submitted),
            "requests_accepted": float(step_requests_accepted),
            "requests_rejected": float(step_requests_rejected),
            "step_fuel_used": float(
                sum(
                    float(stats["fuel_used"])
                    for stats in vessel_step_stats.values()
                )
            ),
            "step_co2_emitted": float(
                sum(
                    float(stats["co2_emitted"])
                    for stats in vessel_step_stats.values()
                )
            ),
            "step_delay_hours": float(sum(step_delay_by_vessel.values())),
        }
        return obs, rewards, done, info

    def sample_stub_actions(self) -> dict[str, Any]:
        """Helper for smoke tests and demos.  Requires ``reset()`` first."""
        if self.medium_forecast is None or self.short_forecast is None:
            raise RuntimeError("call reset() before sample_stub_actions()")
        medium = self.medium_forecast
        short = self.short_forecast

        coordinator_policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        vessel_policy = VesselPolicy(self.cfg, mode="forecast")
        port_policy = PortPolicy(self.cfg, mode="forecast")

        assignments = self._build_assignments()
        directives: list[dict[str, Any]] = []
        for coordinator_id, coordinator in enumerate(self.coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in self.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = self.vessels
            directive = coordinator_policy.propose_action(
                medium_forecast=medium,
                vessels=local_vessels,
                ports=self.ports,
                rng=self.rng,
            )
            directives.append(
                {
                    **directive,
                    "coordinator_id": coordinator_id,
                    "assigned_vessel_ids": local_ids,
                }
            )
        directive_by_vessel: dict[int, dict[str, Any]] = {}
        for directive in directives:
            per_vessel = directive.get("per_vessel_dest", {})
            for vessel_id in directive.get("assigned_vessel_ids", []):
                vessel_dir = dict(directive)
                if vessel_id in per_vessel:
                    vessel_dir["dest_port"] = int(per_vessel[vessel_id])
                directive_by_vessel[vessel_id] = vessel_dir

        vessel_actions = [
            vessel_policy.propose_action(
                short,
                directive_by_vessel.get(
                    vessel_agent.state.vessel_id,
                    self.bus.get_latest_directive(vessel_agent.state.vessel_id)
                    or (directives[0] if directives else self.coordinators[0].last_action),
                ),
            )
            for vessel_agent in self.vessel_agents
        ]
        incoming = sum(1 for a in vessel_actions if a["request_arrival_slot"])
        port_actions = [
            port_policy.propose_action(
                port_agent.state,
                incoming,
                short[i],
            )
            for i, port_agent in enumerate(self.port_agents)
        ]
        return {
            "coordinator": directives[0] if directives else self.coordinators[0].last_action,
            "coordinators": directives,
            "vessels": vessel_actions,
            "ports": port_actions,
        }

    def _refresh_forecasts(self) -> None:
        """Refresh cached forecasts exactly once per environment tick."""
        self.medium_forecast = self.medium_forecaster.predict(self.ports, self.rng)
        self.short_forecast = self.short_forecaster.predict(self.ports, self.rng)

    def _get_observations(self) -> dict[str, Any]:
        """Build observation dicts for all agent types from current state.

        Coordinator observations are **zero-padded** to the maximum
        coordinator dimension (``num_ports * medium_d + num_vessels * 4 + 1``)
        so that parameter-shared networks receive fixed-size inputs even when
        vessel partitions vary.
        """
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast
        if medium is None or short is None:
            raise RuntimeError("forecasts not initialized — call reset() first")

        assignments = self._build_assignments()

        # Fixed coordinator observation dimension
        medium_d = int(self.cfg["medium_horizon_days"])
        max_coord_dim = int(self.cfg["num_ports"]) * medium_d + self.num_vessels * 4 + 1

        coordinator_obs = []
        for coordinator_id, coordinator in enumerate(self.coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in self.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = self.vessels
            raw_obs = coordinator.get_obs(medium, local_vessels)
            # Pad to fixed dimension
            if len(raw_obs) < max_coord_dim:
                raw_obs = np.concatenate(
                    [raw_obs, np.zeros(max_coord_dim - len(raw_obs))]
                )
            coordinator_obs.append(raw_obs[:max_coord_dim])
        coord_obs = coordinator_obs[0]

        vessel_obs = []
        for vessel_agent in self.vessel_agents:
            vessel = vessel_agent.state
            dest = vessel.destination if vessel.destination < self.num_ports else 0
            directive = self.bus.get_latest_directive(vessel.vessel_id)
            if directive is None:
                directive = self._fallback_directive_for_vessel(
                    vessel.vessel_id, assignments
                )
            v_obs = vessel_agent.get_obs(short[dest], directive=directive)
            vessel_obs.append(v_obs)

        port_obs = []
        for i, port_agent in enumerate(self.port_agents):
            p_obs = port_agent.get_obs(
                short[i],
                incoming_requests=len(self.bus.get_pending_requests(i)),
            )
            port_obs.append(p_obs)

        return {
            "coordinator": coord_obs,
            "coordinators": coordinator_obs,
            "vessels": vessel_obs,
            "ports": port_obs,
        }

    def _compute_rewards(
        self,
        vessel_step_stats: dict[int, dict[str, float | bool]] | None = None,
        step_delay_by_vessel: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """Compute per-agent rewards from step-level physics deltas."""
        vessel_step_stats = vessel_step_stats or {}
        step_delay_by_vessel = step_delay_by_vessel or {}
        vessel_rewards = [
            compute_vessel_reward_step(
                vessel=v,
                config=self.cfg,
                fuel_used=float(vessel_step_stats.get(v.vessel_id, {}).get("fuel_used", 0.0)),
                co2_emitted=float(vessel_step_stats.get(v.vessel_id, {}).get("co2_emitted", 0.0)),
                delay_hours=float(step_delay_by_vessel.get(v.vessel_id, 0.0)),
            )
            for v in self.vessels
        ]
        port_rewards = [compute_port_reward(p, self.cfg) for p in self.ports]
        step_fuel_used = float(
            sum(float(stats.get("fuel_used", 0.0)) for stats in vessel_step_stats.values())
        )
        step_co2_emitted = float(
            sum(float(stats.get("co2_emitted", 0.0)) for stats in vessel_step_stats.values())
        )
        coordinator_rewards = [
            compute_coordinator_reward_step(
                ports=self.ports,
                config=self.cfg,
                fuel_used=step_fuel_used,
                co2_emitted=step_co2_emitted,
            )
            for _ in self.coordinators
        ]
        return {
            "coordinator": coordinator_rewards[0],
            "coordinators": coordinator_rewards,
            "vessels": vessel_rewards,
            "ports": port_rewards,
        }

    def get_global_state(self) -> np.ndarray:
        """Flatten all observations and global stats for CTDE critic inputs.

        Coordinator observations are already padded to fixed dimension by
        ``_get_observations()``, so the resulting vector size is deterministic.
        """
        obs = self._get_observations()
        global_congestion = np.array([p.queue for p in self.ports], dtype=float)
        total_emissions = np.array([sum(v.emissions for v in self.vessels)], dtype=float)

        return np.concatenate(
            [
                *obs["coordinators"],
                *(obs["vessels"] if obs["vessels"] else [np.array([])]),
                *(obs["ports"] if obs["ports"] else [np.array([])]),
                global_congestion,
                total_emissions,
            ]
        )

    def _reset_runtime_state(self) -> None:
        """Clear message bus and cache initial port actions."""
        self.bus.reset(self.num_ports)
        self._last_port_actions = [dict(port_agent.last_action) for port_agent in self.port_agents]

    def get_directive_for_vessel(
        self,
        vessel_id: int,
        assignments: dict[int, list[int]] | None = None,
        latest_directive_by_vessel: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return latest directive visible to a vessel, with fallback."""
        assignments = assignments or self._build_assignments()
        latest_map = (
            self.bus.latest_directives
            if latest_directive_by_vessel is None
            else latest_directive_by_vessel
        )
        return latest_map.get(
            vessel_id,
            self._fallback_directive_for_vessel(vessel_id, assignments),
        )

    def peek_step_context(self) -> dict[str, Any]:
        """
        Return the message-visible context at the current step without mutation.

        This mirrors what receivers can observe after latency delivery and before
        new actions are applied in ``step``.
        """
        assignments = self._build_assignments()
        latest_directive_by_vessel, pending_port_requests = self.bus.peek(self.t)
        return {
            "t": self.t,
            "assignments": assignments,
            "latest_directive_by_vessel": latest_directive_by_vessel,
            "pending_port_requests": pending_port_requests,
        }

    def _build_assignments(self) -> dict[int, list[int]]:
        """Partition vessels across coordinators for the current step."""
        return assign_vessels_to_coordinators(self.vessels, self.num_coordinators)

    def _fallback_directive_for_vessel(
        self,
        vessel_id: int,
        assignments: dict[int, list[int]] | None = None,
    ) -> dict[str, Any]:
        """Return a default directive when no message has been delivered yet."""
        assignments = assignments or self._build_assignments()
        for coordinator_id, local_ids in assignments.items():
            if vessel_id in local_ids:
                action = dict(self.coordinators[coordinator_id].last_action)
                action["coordinator_id"] = coordinator_id
                action["assigned_vessel_ids"] = list(local_ids)
                return action
        action = dict(self.coordinators[0].last_action)
        action["coordinator_id"] = 0
        action["assigned_vessel_ids"] = []
        return action

    def _normalize_coordinator_actions(self, actions: dict[str, Any]) -> list[dict[str, Any]]:
        """Ensure coordinator actions are a list matching coordinator count."""
        raw = actions.get("coordinators")
        if raw is None:
            raw = [actions.get("coordinator", {})]
        if not isinstance(raw, list):
            raw = [raw]
        if not raw:
            raw = [{}]
        normalized: list[dict[str, Any]] = []
        for coordinator_id in range(self.num_coordinators):
            if coordinator_id < len(raw):
                candidate = raw[coordinator_id]
            else:
                candidate = raw[-1]
            if not isinstance(candidate, dict):
                candidate = {}
            normalized.append(candidate)
        return normalized

    def _normalize_vessel_actions(self, actions: dict[str, Any]) -> list[dict[str, Any]]:
        """Ensure vessel actions are a list matching vessel count."""
        raw = actions.get("vessels", [])
        if not isinstance(raw, list):
            raw = []
        normalized: list[dict[str, Any]] = []
        for vessel_id in range(self.num_vessels):
            if vessel_id < len(raw) and isinstance(raw[vessel_id], dict):
                normalized.append(raw[vessel_id])
            else:
                normalized.append(dict(self.vessel_agents[vessel_id].last_action))
        return normalized

    def _normalize_port_actions(self, actions: dict[str, Any]) -> list[dict[str, Any]]:
        """Ensure port actions are a list matching port count."""
        raw = actions.get("ports", [])
        if not isinstance(raw, list):
            raw = []
        normalized: list[dict[str, Any]] = []
        for port_id in range(self.num_ports):
            if port_id < len(raw) and isinstance(raw[port_id], dict):
                normalized.append(raw[port_id])
            elif self._last_port_actions:
                normalized.append(dict(self._last_port_actions[port_id]))
            else:
                normalized.append(dict(self.port_agents[port_id].last_action))
        return normalized

