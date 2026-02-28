"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent, assign_vessels_to_coordinators
from .config import SEED, DecisionCadence, get_default_config, resolve_distance_matrix
from .dynamics import (
    dispatch_vessel,
    generate_weather,
    step_ports,
    step_vessels,
    update_weather_ar1,
)
from .forecasts import MediumTermForecaster, ShortTermForecaster
from .message_bus import MessageBus
from .metrics import compute_port_metrics
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_vessel_reward_step,
    weather_coordinator_shaping,
    weather_vessel_shaping,
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
        self._weather_enabled = bool(self.cfg.get("weather_enabled", False))
        self._weather: np.ndarray | None = None

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
        self._refresh_weather()
        self._refresh_forecasts()
        return self._get_observations()

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool, dict[str, Any]]:
        """Execute one environment tick.

        Implements the joint transition kernel T(s' | s, a) in five ordered
        phases:

        Phase 0 – Message delivery
            ``bus.deliver_due(t)`` fires all messages whose delivery step has
            arrived: coordinator directives land in each vessel's mailbox;
            vessel arrival requests land in port pending queues; port slot
            responses are returned for processing in phase 2.

        Phase 1 – Coordinator action
            Each coordinator (on its sub-cadence) converts its action dict into
            per-vessel directives enqueued on the bus with latency.  Directives
            are visible to vessels only after delivery (phase 0 of a future tick).

        Phase 2 – Vessel action + slot negotiation
            Each vessel (on its sub-cadence) decides speed and whether to request
            a berth slot.  If a slot response from phase 0 was *accepted*,
            ``dispatch_vessel()`` commits ``vessel.destination``, ``vessel.at_sea``,
            and ``vessel.position_nm = 0``.  Rejected or still-waiting vessels
            accumulate ``delay_hours``.

        Phase 3 – Physics tick
            ``step_vessels()`` advances every in-transit vessel by one time step:
            ``position_nm += speed * dt * weather_speed_factor(W_t)``.
            Vessels that complete their leg flip ``at_sea = False`` and are
            appended to their destination port's queue.

        Phase 4 – Port action + service tick
            Each port (on its sub-cadence) allocates berth slots from its pending
            request backlog and enqueues accept/reject responses.  ``step_ports()``
            drains berths whose service time has elapsed and admits queued vessels.

        Phase 5 – Reward + clock advance
            Rewards are computed from the state produced by phases 1–4 using
            weather ``W_t`` (active during this tick).  Only then does the clock
            advance (``t += 1``) and weather update to ``W_{t+1}`` via AR(1).
            The observations returned therefore reflect ``W_{t+1}``; agents
            should treat the weather in ``obs`` as the forecast for their
            *next* action, not the one that generated the current reward.

        Parameters
        ----------
        actions:
            Dict with keys ``"coordinator"`` (single dict or empty),
            ``"coordinators"`` (list of dicts), ``"vessels"`` (list of
            dicts with ``"target_speed"`` and ``"request_arrival_slot"``),
            ``"ports"`` (list of dicts with ``"service_rate"`` and
            ``"accept_requests"``).

        Returns
        -------
        obs : dict
            ``{"vessels": [...], "ports": [...], "coordinators": [...]}``.
            Weather in obs reflects ``W_{t+1}`` (post-update).
        rewards : dict
            ``{"vessels": [...], "ports": [...], "coordinator": float,
            "coordinators": [...]}``.
        done : bool
            True when ``t >= rollout_steps``.
        info : dict
            Step-level metadata including ``"port_metrics"``,
            ``"cadence_due"``, ``"requests_submitted"``, etc.
        """
        # ── Phase 0: message delivery ─────────────────────────────────────────
        # Fire all messages whose deliver_step <= t.  Directives land in each
        # vessel's mailbox; arrival requests land in port pending queues;
        # slot responses are returned as delivered_responses for phase 2.
        due = self.cadence.due(self.t)
        delivered_responses = self.bus.deliver_due(self.t)
        assignments = self._build_assignments()
        step_delay_by_vessel: dict[int, float] = {
            vessel.vessel_id: 0.0 for vessel in self.vessels
        }
        step_requests_submitted = 0
        step_requests_accepted = 0
        step_requests_rejected = 0

        # ── Phase 1: coordinator action ──────────────────────────────────────
        # Each active coordinator converts its action into per-vessel directives
        # and enqueues them with message_latency_steps delay.  Vessels will read
        # these directives only after they are delivered in a future phase 0.
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

        # ── Phase 2: vessel action + slot negotiation ────────────────────────
        # Each vessel reads its latest directive, decides speed and whether to
        # request a berth slot.  If a slot response delivered in phase 0 is
        # accepted, dispatch_vessel() commits the destination and sets at_sea.
        # Rejected or still-waiting vessels accrue delay_hours.
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
                    and not vessel.pending_departure
                    and not self.bus.is_awaiting(vessel_id)
                ):
                    destination = int(directive.get("dest_port", vessel.destination))
                    self.bus.enqueue_arrival_request(
                        self.t + self.cadence.message_latency_steps,
                        vessel_id,
                        destination,
                        requested_arrival_time=float(normalized.get("requested_arrival_time", 0.0)),
                    )
                    self.bus.mark_awaiting(vessel_id)
                    step_requests_submitted += 1
            else:
                normalized = dict(vessel_agent.last_action)

            response = delivered_responses.get(vessel_id)
            if response is not None:
                if response["accepted"] and not vessel.at_sea:
                    dep_window = int(directive.get("departure_window_hours", 0))
                    dispatch_vessel(
                        vessel=vessel,
                        destination=int(response["dest_port"]),
                        speed=float(normalized["target_speed"]),
                        config=self.cfg,
                        current_step=self.t,
                        departure_window_hours=dep_window,
                        dt_hours=float(self.cfg.get("dt_hours", 1.0)),
                    )
                elif not response["accepted"] and not vessel.at_sea:
                    vessel.delay_hours += 1.0
                    step_delay_by_vessel[vessel_id] += 1.0
                self.bus.clear_awaiting(vessel_id)
            elif self.bus.is_awaiting(vessel_id) and not vessel.at_sea:
                vessel.delay_hours += 1.0
                step_delay_by_vessel[vessel_id] += 1.0
            normalized_vessel_actions.append(normalized)

        # ── Phase 3: physics tick ────────────────────────────────────────────
        # Advance all in-transit vessels using weather W_t (current tick).
        # position_nm += speed * dt * weather_speed_factor(W_t).
        # Vessels that complete their leg are appended to the destination
        # port's queue and flipped to at_sea = False.
        pre_step_at_sea = {v.vessel_id: v.at_sea for v in self.vessels}

        vessel_step_stats = step_vessels(
            vessels=self.vessels,
            distance_nm=self.distance_nm,
            config=self.cfg,
            dt_hours=1.0,
            weather=self._weather if self._weather_enabled else None,
            current_step=self.t,
        )

        for vessel in self.vessels:
            if (
                pre_step_at_sea.get(vessel.vessel_id, False)
                and not vessel.at_sea
                and vessel.location == vessel.destination
            ):
                self.ports[vessel.location].queue += 1

        # ── Phase 4: port action + service tick ──────────────────────────────
        # Ports (on their sub-cadence) accept/reject pending berth requests and
        # enqueue slot responses.  step_ports() then drains completed berths,
        # admits queued vessels, and accumulates per-port wait-time statistics.
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

        # ── Phase 5: reward computation ──────────────────────────────────────
        # Rewards are computed from the state produced by phases 1–4, using
        # weather W_t (the value active during this tick).  The clock and
        # weather are advanced AFTER rewards so that r_t = R(s_t, a_t, W_t).
        rewards = self._compute_rewards(
            vessel_step_stats=vessel_step_stats,
            step_delay_by_vessel=step_delay_by_vessel,
        )

        # ── Advance clock and environment stochastic state ───────────────────
        # t increments here; _refresh_weather() advances W_t → W_{t+1} via
        # AR(1).  Observations returned below therefore contain W_{t+1}, which
        # agents should treat as the weather forecast for their *next* action.
        self.t += 1
        done = self.t >= self.cfg["rollout_steps"]
        self._refresh_weather()
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
            "weather_enabled": self._weather_enabled,
        }
        if self._weather_enabled and self._weather is not None:
            info["mean_sea_state"] = float(np.mean(self._weather))
            info["max_sea_state"] = float(np.max(self._weather))
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

    def _refresh_weather(self) -> None:
        """Generate or update per-route sea-state matrix for this tick.

        Uses AR(1) temporal correlation when ``weather_autocorrelation > 0``
        and a previous weather matrix exists; otherwise falls back to i.i.d.
        sampling.
        """
        if not self._weather_enabled:
            self._weather = None
            return
        sea_state_max = float(self.cfg.get("sea_state_max", 3.0))
        autocorr = float(self.cfg.get("weather_autocorrelation", 0.0))
        if autocorr > 0.0 and self._weather is not None:
            self._weather = update_weather_ar1(
                self._weather, self.rng, autocorr, sea_state_max
            )
        else:
            self._weather = generate_weather(self.num_ports, self.rng, sea_state_max)

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
            directive = self.bus.get_latest_directive(vessel.vessel_id)
            if directive is None:
                directive = self._fallback_directive_for_vessel(
                    vessel.vessel_id, assignments
                )
            # For vessels at sea, the navigation destination is correct.
            # For docked/pending vessels the coordinator directive's dest_port is
            # the relevant port — vessel.destination may be stale (last arrival).
            # Using the directive keeps the forecast aligned with the FC assignment
            # even across FC reassignments during the message-latency window.
            if vessel.at_sea:
                obs_dest = vessel.destination if vessel.destination < self.num_ports else 0
            else:
                obs_dest = int(directive.get("dest_port", vessel.destination))
                obs_dest = obs_dest if 0 <= obs_dest < self.num_ports else 0

            dest = obs_dest  # kept for dock_avail and weather indexing below
            # Dock availability fraction at destination port (proposal §4.1)
            dest_port = self.ports[dest]
            dock_avail = max(dest_port.docks - dest_port.occupied, 0) / max(
                dest_port.docks, 1
            )
            v_obs = vessel_agent.get_obs(
                short[dest], directive=directive, dock_availability=dock_avail,
            )
            # Append weather sea state for the vessel's current route
            if self._weather_enabled and self._weather is not None:
                src = vessel.location
                sea_state = float(self._weather[src, dest])
                v_obs = np.concatenate([v_obs, np.array([sea_state])])
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
        # Apply weather shaping bonus for vessels that slow in rough seas.
        # Only at-sea vessels receive shaping — docked vessels make no routing decisions.
        if self._weather_enabled and self._weather is not None:
            for i, v in enumerate(self.vessels):
                if not v.at_sea:
                    continue
                speed = float(v.speed)
                src, dst = v.location, v.destination
                n = self._weather.shape[0]
                sea = float(self._weather[src, dst]) if 0 <= src < n and 0 <= dst < n else 0.0
                vessel_rewards[i] += weather_vessel_shaping(speed, sea, self.cfg)
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
        # Apply weather shaping bonus for coordinators routing through calm seas.
        if self._weather_enabled and self._weather is not None:
            routes: list[tuple[int, int]] = []
            for v in self.vessels:
                d = self.bus.get_latest_directive(v.vessel_id)
                if d:
                    routes.append((v.location, int(d.get("dest_port", v.destination))))
            if routes:
                bonus = weather_coordinator_shaping(self._weather, routes, self.cfg)
                coordinator_rewards = [r + bonus for r in coordinator_rewards]
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

