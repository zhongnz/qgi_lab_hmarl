"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .config import DEFAULT_CONFIG, DISTANCE_NM, SEED, get_default_config
from .dynamics import dispatch_vessel, observe_port_metrics, step_ports, step_vessels
from .forecasts import MediumTermForecaster, ShortTermForecaster
from .multi_coordinator import assign_vessels_to_coordinators
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward,
    compute_port_reward,
    compute_vessel_reward,
)
from .scheduling import DecisionCadence
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
        self.distance_nm = DISTANCE_NM if distance_nm is None else distance_nm
        self.cadence = DecisionCadence.from_config(self.cfg)
        self.t = 0

        self.ports: list[PortState] = []
        self.vessels: list[VesselState] = []
        self.coordinators: list[FleetCoordinatorAgent] = []
        self.coordinator: FleetCoordinatorAgent | None = None
        self.port_agents: list[PortAgent] = []
        self.vessel_agents: list[VesselAgent] = []
        self.medium_forecast: np.ndarray | None = None
        self.short_forecast: np.ndarray | None = None
        self.medium_forecaster = MediumTermForecaster(self.cfg["medium_horizon_days"])
        self.short_forecaster = ShortTermForecaster(self.cfg["short_horizon_hours"])
        self.coordinator_policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        self.vessel_policy = VesselPolicy(self.cfg, mode="forecast")
        self.port_policy = PortPolicy(self.cfg, mode="forecast")

        self.obs_shapes = {
            "coordinator": (
                self.num_ports * self.cfg["medium_horizon_days"]
                + self.num_vessels * 4
                + 1
            ),
            "vessel": 1 + 1 + 1 + 1 + self.cfg["short_horizon_hours"] + 3,
            "port": 1 + 1 + 1 + self.cfg["short_horizon_hours"] + 1,
        }
        self.action_shapes = {
            "coordinator": self.num_ports + 2,
            "vessel": 2,
            "port": 2,
        }
        self._directive_queue: list[tuple[int, int, dict[str, Any]]] = []
        self._arrival_request_queue: list[tuple[int, int, int]] = []
        self._slot_response_queue: list[tuple[int, int, bool, int]] = []
        self._pending_port_requests: dict[int, list[int]] = {
            port_id: [] for port_id in range(self.num_ports)
        }
        self._awaiting_slot_response: set[int] = set()
        self._latest_directive_by_vessel: dict[int, dict[str, Any]] = {}
        self._last_port_actions: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        """Reset state and return initial observations."""
        self.t = 0
        self.rng = make_rng(self.seed)
        self.ports = initialize_ports(
            num_ports=self.num_ports,
            docks_per_port=self.cfg["docks_per_port"],
            rng=self.rng,
        )
        self.vessels = initialize_vessels(
            num_vessels=self.num_vessels,
            num_ports=self.num_ports,
            nominal_speed=self.cfg["nominal_speed"],
            rng=self.rng,
        )
        self.coordinators = [
            FleetCoordinatorAgent(config=self.cfg, coordinator_id=i)
            for i in range(self.num_coordinators)
        ]
        self.coordinator = self.coordinators[0]
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
        delivered_responses = self._deliver_due_messages()
        assignments = self._build_assignments()

        coordinator_actions = self._normalize_coordinator_actions(actions)
        if due["coordinator"]:
            for coordinator_id, coordinator in enumerate(self.coordinators):
                applied = coordinator.apply_action(coordinator_actions[coordinator_id])
                enriched = {
                    **applied,
                    "coordinator_id": coordinator_id,
                    "assigned_vessel_ids": assignments.get(coordinator_id, []),
                }
                for vessel_id in assignments.get(coordinator_id, []):
                    self._directive_queue.append(
                        (
                            self.t + self.cadence.message_latency_steps,
                            vessel_id,
                            dict(enriched),
                        )
                    )

        vessel_inputs = self._normalize_vessel_actions(actions)
        normalized_vessel_actions: list[dict[str, Any]] = []
        for idx, vessel_agent in enumerate(self.vessel_agents):
            vessel = vessel_agent.state
            vessel_id = vessel.vessel_id
            directive = self._latest_directive_by_vessel.get(
                vessel_id,
                self._fallback_directive_for_vessel(vessel_id, assignments),
            )

            if due["vessel"]:
                normalized = vessel_agent.apply_action(vessel_inputs[idx])
                if (
                    normalized.get("request_arrival_slot", False)
                    and not vessel.at_sea
                    and vessel_id not in self._awaiting_slot_response
                ):
                    destination = int(directive.get("dest_port", vessel.destination))
                    self._arrival_request_queue.append(
                        (self.t + self.cadence.message_latency_steps, vessel_id, destination)
                    )
                    self._awaiting_slot_response.add(vessel_id)
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
                self._awaiting_slot_response.discard(vessel_id)
            elif vessel_id in self._awaiting_slot_response and not vessel.at_sea:
                vessel.delay_hours += 1.0
            normalized_vessel_actions.append(normalized)

        pre_step_at_sea = {v.vessel_id: v.at_sea for v in self.vessels}

        step_vessels(
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
                backlog = self._pending_port_requests.get(port_id, [])
                available_slots = max(port_agent.state.docks - port_agent.state.occupied, 0)
                accept_limit = min(
                    len(backlog),
                    max(int(normalized.get("accept_requests", 0)), 0),
                    available_slots,
                )
                accepted = backlog[:accept_limit]
                rejected = backlog[accept_limit:]
                self._pending_port_requests[port_id] = []
                response_step = self.t + self.cadence.message_latency_steps
                for vessel_id in accepted:
                    self._slot_response_queue.append((response_step, vessel_id, True, port_id))
                for vessel_id in rejected:
                    self._slot_response_queue.append((response_step, vessel_id, False, port_id))
            self._last_port_actions = normalized_port_actions

        service_rates = [
            int(action.get("service_rate", 1))
            for action in (
                self._last_port_actions
                if self._last_port_actions
                else [port_agent.last_action for port_agent in self.port_agents]
            )
        ]
        step_ports(self.ports, service_rates, dt_hours=1.0)

        rewards = self._compute_rewards()

        self.t += 1
        done = self.t >= self.cfg["rollout_steps"]
        self._refresh_forecasts()
        obs = self._get_observations()
        info = {
            "port_metrics": observe_port_metrics(self.ports),
            "cadence_due": due,
            "coordinator_assignments": assignments,
            "message_queues": {
                "directives": len(self._directive_queue),
                "arrival_requests": len(self._arrival_request_queue),
                "slot_responses": len(self._slot_response_queue),
            },
        }
        return obs, rewards, done, info

    def sample_stub_actions(self) -> dict[str, Any]:
        """Helper for smoke tests and demos."""
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast
        if not self.coordinators:
            self.coordinators = [
                FleetCoordinatorAgent(config=self.cfg, coordinator_id=i)
                for i in range(self.num_coordinators)
            ]
            self.coordinator = self.coordinators[0]

        assignments = self._build_assignments()
        directives: list[dict[str, Any]] = []
        for coordinator_id, coordinator in enumerate(self.coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in self.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = self.vessels
            directive = self.coordinator_policy.act(
                agent=coordinator,
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
        directive_by_vessel = {}
        for directive in directives:
            for vessel_id in directive.get("assigned_vessel_ids", []):
                directive_by_vessel[vessel_id] = directive

        vessel_actions = [
            self.vessel_policy.act(
                vessel_agent,
                short,
                directive_by_vessel.get(
                    vessel_agent.state.vessel_id,
                    self._latest_directive_by_vessel.get(
                        vessel_agent.state.vessel_id,
                        directives[0] if directives else self.coordinators[0].last_action,
                    ),
                ),
            )
            for vessel_agent in self.vessel_agents
        ]
        incoming = sum(1 for a in vessel_actions if a["request_arrival_slot"])
        port_actions = [
            self.port_policy.act(port_agent, incoming, short[i])
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
        self.medium_forecast = self.medium_forecaster.predict(self.num_ports, self.rng)
        self.short_forecast = self.short_forecaster.predict(self.num_ports, self.rng)

    def _get_observations(self) -> dict[str, Any]:
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast

        if not self.coordinators:
            self.coordinators = [
                FleetCoordinatorAgent(config=self.cfg, coordinator_id=i)
                for i in range(self.num_coordinators)
            ]
            self.coordinator = self.coordinators[0]
        assignments = self._build_assignments()
        coordinator_obs = []
        for coordinator_id, coordinator in enumerate(self.coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in self.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = self.vessels
            coordinator_obs.append(coordinator.get_obs(medium, local_vessels))
        coord_obs = coordinator_obs[0]

        vessel_obs = []
        for vessel_agent in self.vessel_agents:
            vessel = vessel_agent.state
            dest = vessel.destination if vessel.destination < self.num_ports else 0
            directive = self._latest_directive_by_vessel.get(
                vessel.vessel_id,
                self._fallback_directive_for_vessel(vessel.vessel_id, assignments),
            )
            v_obs = vessel_agent.get_obs(short[dest], directive=directive)
            vessel_obs.append(v_obs)

        port_obs = []
        for i, port_agent in enumerate(self.port_agents):
            p_obs = port_agent.get_obs(
                short[i],
                incoming_requests=len(self._pending_port_requests.get(i, [])),
            )
            port_obs.append(p_obs)

        return {
            "coordinator": coord_obs,
            "coordinators": coordinator_obs,
            "vessels": vessel_obs,
            "ports": port_obs,
        }

    def _compute_rewards(self) -> dict[str, Any]:
        vessel_rewards = [compute_vessel_reward(v, self.cfg) for v in self.vessels]
        port_rewards = [compute_port_reward(p, self.cfg) for p in self.ports]
        coordinator_rewards = [
            compute_coordinator_reward(self.vessels, self.ports, self.cfg)
            for _ in self.coordinators
        ]
        return {
            "coordinator": coordinator_rewards[0],
            "coordinators": coordinator_rewards,
            "vessels": vessel_rewards,
            "ports": port_rewards,
        }

    def get_global_state(self) -> np.ndarray:
        """Flatten all observations and global stats for CTDE critic inputs."""
        obs = self._get_observations()
        global_congestion = np.array([p.queue for p in self.ports], dtype=float)
        total_emissions = np.array([sum(v.emissions for v in self.vessels)], dtype=float)
        extra_coordinators = (
            obs["coordinators"][1:]
            if obs.get("coordinators") and len(obs["coordinators"]) > 1
            else []
        )
        return np.concatenate(
            [
                obs["coordinator"],
                *(extra_coordinators if extra_coordinators else [np.array([])]),
                *(obs["vessels"] if obs["vessels"] else [np.array([])]),
                *(obs["ports"] if obs["ports"] else [np.array([])]),
                global_congestion,
                total_emissions,
            ]
        )

    def _reset_runtime_state(self) -> None:
        self._directive_queue = []
        self._arrival_request_queue = []
        self._slot_response_queue = []
        self._pending_port_requests = {port_id: [] for port_id in range(self.num_ports)}
        self._awaiting_slot_response = set()
        self._latest_directive_by_vessel = {}
        self._last_port_actions = [dict(port_agent.last_action) for port_agent in self.port_agents]

    def _build_assignments(self) -> dict[int, list[int]]:
        return assign_vessels_to_coordinators(self.vessels, self.num_coordinators)

    def _fallback_directive_for_vessel(
        self,
        vessel_id: int,
        assignments: dict[int, list[int]] | None = None,
    ) -> dict[str, Any]:
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

    def _deliver_due_messages(self) -> dict[int, dict[str, Any]]:
        delivered_responses: dict[int, dict[str, Any]] = {}

        remaining_directives: list[tuple[int, int, dict[str, Any]]] = []
        for deliver_step, vessel_id, directive in self._directive_queue:
            if deliver_step <= self.t:
                self._latest_directive_by_vessel[vessel_id] = directive
            else:
                remaining_directives.append((deliver_step, vessel_id, directive))
        self._directive_queue = remaining_directives

        remaining_requests: list[tuple[int, int, int]] = []
        for deliver_step, vessel_id, destination in self._arrival_request_queue:
            if deliver_step <= self.t:
                if 0 <= destination < self.num_ports:
                    self._pending_port_requests[destination].append(vessel_id)
            else:
                remaining_requests.append((deliver_step, vessel_id, destination))
        self._arrival_request_queue = remaining_requests

        remaining_responses: list[tuple[int, int, bool, int]] = []
        for deliver_step, vessel_id, accepted, destination in self._slot_response_queue:
            if deliver_step <= self.t:
                delivered_responses[vessel_id] = {
                    "accepted": bool(accepted),
                    "dest_port": int(destination),
                }
            else:
                remaining_responses.append((deliver_step, vessel_id, accepted, destination))
        self._slot_response_queue = remaining_responses

        return delivered_responses


def make_default_env() -> MaritimeEnv:
    """Convenience factory used by scripts and tests."""
    return MaritimeEnv(config=DEFAULT_CONFIG)
