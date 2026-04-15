"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent, assign_vessels_to_coordinators
from .config import SEED, DecisionCadence, get_default_config, resolve_distance_matrix
from .dynamics import (
    compute_fuel_and_emissions,
    dispatch_vessel,
    generate_weather,
    step_ports,
    step_vessels,
    update_weather_ar1,
    weather_speed_factor,
)
from .forecasts import (
    GroundTruthForecaster,
    MediumTermForecaster,
    NoiselessForecaster,
    ShortTermForecaster,
)
from .message_bus import MessageBus
from .metrics import compute_port_metrics
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward_breakdown,
    compute_port_reward_breakdown,
    compute_vessel_reward_breakdown,
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
        self._next_reset_seed = self.seed
        self._last_reset_seed = self.seed
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
        self.noiseless_forecaster = NoiselessForecaster(
            medium_horizon_days=self.cfg["medium_horizon_days"],
            short_horizon_hours=self.cfg["short_horizon_hours"],
        )
        self.ground_truth_forecaster = GroundTruthForecaster(
            medium_horizon_days=self.cfg["medium_horizon_days"],
            short_horizon_hours=self.cfg["short_horizon_hours"],
            config=self.cfg,
            distance_nm=self.distance_nm,
        )

        self.bus = MessageBus(self.num_ports)
        self._last_port_actions: list[dict[str, Any]] = []
        self._weather_enabled = bool(self.cfg.get("weather_enabled", True))
        self._weather: np.ndarray | None = None
        self._last_done_reason = "rollout_limit"

    def _single_mission_enabled(self) -> bool:
        """Return True when the episodic single-mission variant is active."""
        return str(self.cfg.get("episode_mode", "continuous")) == "single_mission"

    def _mission_success_on(self) -> str:
        """Return the terminal-success criterion for single-mission mode."""
        return str(self.cfg.get("mission_success_on", "arrival"))

    def _mark_vessel_mission_terminal(
        self,
        vessel: VesselState,
        *,
        success: bool,
    ) -> None:
        """Mark a vessel's single mission as complete and suppress redispatch."""
        vessel.mission_done = True
        vessel.mission_success = bool(success)
        vessel.mission_failed = not bool(success)
        vessel.pending_departure = False
        vessel.pending_requested_arrival_time = 0.0
        vessel.requested_arrival_time = 0.0

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Reset state and return initial observations.

        Reset seeding follows two simple rules:

        - ``reset(seed=123)`` is fully deterministic and replays the exact
          same initial state whenever the same seed is provided.
        - ``reset()`` advances to the next reproducible episode seed so that
          repeated training resets explore different initial states while
          remaining deterministic for a fixed run seed.
        """
        if seed is not None:
            effective_seed = int(seed)
            self.seed = effective_seed
            self._next_reset_seed = effective_seed + 1
        elif self.seed != self._next_reset_seed - 1:
            # Respect explicit ``env.seed = ...`` overrides used by older code.
            effective_seed = int(self.seed)
            self._next_reset_seed = effective_seed + 1
        else:
            effective_seed = int(self._next_reset_seed)
            self.seed = effective_seed
            self._next_reset_seed = effective_seed + 1

        self.t = 0
        self._last_reset_seed = effective_seed
        self.rng = make_rng(effective_seed)
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
        self._sync_vessel_port_service_state()
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
            dicts with ``"target_speed"``, ``"request_arrival_slot"``,
            and optional ``"requested_arrival_time"``),
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
        # ── Setup ─────────────────────────────────────────────────────────────
        # Per-step bookkeeping: cadence flags, coordinator assignments, counters.
        due = self.cadence.due(self.t)
        assignments = self._build_assignments()
        dt_hours = float(self.cfg.get("dt_hours", 1.0))
        self._sync_vessel_port_service_state()
        step_delay_by_vessel: dict[int, float] = {
            vessel.vessel_id: 0.0 for vessel in self.vessels
        }
        step_schedule_delay_by_vessel: dict[int, float] = {
            vessel.vessel_id: 0.0 for vessel in self.vessels
        }
        step_on_time_arrival_by_vessel: dict[int, bool] = {
            vessel.vessel_id: False for vessel in self.vessels
        }
        step_requests_submitted = 0
        step_requests_accepted = 0
        step_requests_rejected = 0
        step_on_time_arrivals = 0.0
        step_scheduled_arrivals_completed = 0.0
        step_fuel_capped_departures = 0.0
        step_fuel_blocked_departures = 0.0
        step_refueled_vessels = 0.0
        step_mission_successes = 0.0
        step_mission_failures = 0.0
        step_events: list[dict[str, Any]] = []

        def log_event(event_type: str, stage: str, **fields: Any) -> None:
            event = {
                "t": int(self.t),
                "stage": stage,
                "event_type": event_type,
            }
            event.update(fields)
            step_events.append(event)

        # ── Phase 0: message delivery ─────────────────────────────────────────
        # Fire all messages whose deliver_step <= t.  Directives land in each
        # vessel's mailbox; arrival requests land in port pending queues;
        # slot responses are returned as delivered_responses for phase 2.
        delivered_responses = self.bus.deliver_due(self.t)
        step_events.extend(self.bus.last_delivery_events)

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
                    log_event(
                        "directive_enqueued",
                        "phase1",
                        coordinator_id=int(coordinator_id),
                        vessel_id=int(vessel_id),
                        port_id=int(vessel_directive.get("dest_port", -1)),
                        deliver_step=int(self.t + self.cadence.message_latency_steps),
                        departure_window_hours=int(
                            vessel_directive.get("departure_window_hours", 0)
                        ),
                        emission_budget=float(vessel_directive.get("emission_budget", 0.0)),
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

            if self._single_mission_enabled() and bool(vessel.mission_done):
                self.bus.clear_awaiting(vessel_id)
                normalized = {
                    "target_speed": float(vessel.speed),
                    "request_arrival_slot": False,
                    "requested_arrival_time": 0.0,
                }
            elif due["vessel"]:
                normalized = vessel_agent.apply_action(vessel_inputs[idx])
                if (
                    normalized.get("request_arrival_slot", False)
                    and not vessel.at_sea
                    and not vessel.pending_departure
                    and not self._is_vessel_port_busy(vessel_id)
                    and not self.bus.is_awaiting(vessel_id)
                ):
                    destination = int(directive.get("dest_port", vessel.destination))
                    dep_window = int(directive.get("departure_window_hours", 0))
                    resolved_arrival_time = self._resolve_requested_arrival_time(
                        vessel,
                        destination,
                        float(normalized.get("requested_arrival_time", 0.0)),
                        departure_window_hours=dep_window,
                    )
                    vessel.pending_requested_arrival_time = resolved_arrival_time
                    self.bus.enqueue_arrival_request(
                        self.t + self.cadence.message_latency_steps,
                        vessel_id,
                        destination,
                        requested_arrival_time=resolved_arrival_time,
                    )
                    log_event(
                        "arrival_request_enqueued",
                        "phase2",
                        vessel_id=int(vessel_id),
                        port_id=int(destination),
                        deliver_step=int(self.t + self.cadence.message_latency_steps),
                        requested_arrival_time=float(resolved_arrival_time),
                        requested_speed=float(normalized.get("target_speed", vessel.speed)),
                    )
                    self.bus.mark_awaiting(vessel_id)
                    step_requests_submitted += 1
            else:
                normalized = dict(vessel_agent.last_action)

            response = delivered_responses.get(vessel_id)
            if response is not None:
                if (
                    response["accepted"]
                    and not vessel.at_sea
                    and not self._is_vessel_port_busy(vessel_id)
                ):
                    dep_window = int(directive.get("departure_window_hours", 0))
                    dispatch_speed, speed_capped = self._resolve_fuel_safe_dispatch_speed(
                        vessel,
                        int(response["dest_port"]),
                        float(normalized["target_speed"]),
                    )
                    if dispatch_speed is None:
                        vessel.delay_hours += dt_hours
                        step_delay_by_vessel[vessel_id] += dt_hours
                        vessel.pending_requested_arrival_time = 0.0
                        step_fuel_blocked_departures += 1.0
                        log_event(
                            "dispatch_blocked_fuel",
                            "phase2",
                            vessel_id=int(vessel_id),
                            port_id=int(response["dest_port"]),
                            requested_speed=float(normalized["target_speed"]),
                            available_fuel=float(vessel.fuel),
                        )
                        if self._single_mission_enabled():
                            self._mark_vessel_mission_terminal(vessel, success=False)
                            step_mission_failures += 1.0
                            log_event(
                                "mission_failed",
                                "phase2",
                                vessel_id=int(vessel_id),
                                reason="dispatch_blocked_fuel",
                            )
                    else:
                        if speed_capped:
                            step_fuel_capped_departures += 1.0
                            log_event(
                                "dispatch_speed_capped",
                                "phase2",
                                vessel_id=int(vessel_id),
                                port_id=int(response["dest_port"]),
                                requested_speed=float(normalized["target_speed"]),
                                applied_speed=float(dispatch_speed),
                                available_fuel=float(vessel.fuel),
                            )
                        dispatch_vessel(
                            vessel=vessel,
                            destination=int(response["dest_port"]),
                            speed=float(dispatch_speed),
                            config=self.cfg,
                            current_step=self.t,
                            departure_window_hours=dep_window,
                            dt_hours=dt_hours,
                            requested_arrival_time=float(vessel.pending_requested_arrival_time),
                        )
                        log_event(
                            "vessel_dispatched",
                            "phase2",
                            vessel_id=int(vessel_id),
                            port_id=int(response["dest_port"]),
                            applied_speed=float(dispatch_speed),
                            departure_window_hours=int(dep_window),
                            requested_arrival_time=float(vessel.requested_arrival_time),
                        )
                elif not response["accepted"] and not vessel.at_sea:
                    vessel.delay_hours += dt_hours
                    step_delay_by_vessel[vessel_id] += dt_hours
                    vessel.pending_requested_arrival_time = 0.0
                self.bus.clear_awaiting(vessel_id)
            elif self.bus.is_awaiting(vessel_id) and not vessel.at_sea:
                vessel.delay_hours += dt_hours
                step_delay_by_vessel[vessel_id] += dt_hours
            normalized_vessel_actions.append(normalized)

        # ── Phase 3: physics tick ────────────────────────────────────────────
        # Advance all in-transit vessels using weather W_t (current tick).
        # position_nm += speed * dt * weather_speed_factor(W_t).
        # Vessels that complete their leg are appended to the destination
        # port's queue and flipped to at_sea = False.
        pre_step_at_sea = {v.vessel_id: v.at_sea for v in self.vessels}
        pre_step_trip_active = {
            v.vessel_id: bool(v.at_sea or v.pending_departure) for v in self.vessels
        }
        pre_step_deadline = {
            v.vessel_id: float(v.requested_arrival_time) for v in self.vessels
        }

        vessel_step_stats = step_vessels(
            vessels=self.vessels,
            distance_nm=self.distance_nm,
            config=self.cfg,
            dt_hours=dt_hours,
            weather=self._weather if self._weather_enabled else None,
            current_step=self.t,
        )

        for vessel in self.vessels:
            stall_hours = float(
                vessel_step_stats.get(vessel.vessel_id, {}).get("stall_hours", 0.0)
            )
            if stall_hours > 0.0:
                vessel.delay_hours += stall_hours
                step_delay_by_vessel[vessel.vessel_id] += stall_hours
                if self._single_mission_enabled() and not vessel.mission_done:
                    self._mark_vessel_mission_terminal(vessel, success=False)
                    step_mission_failures += 1.0
                    log_event(
                        "mission_failed",
                        "phase3",
                        vessel_id=int(vessel.vessel_id),
                        reason="fuel_stall",
                    )
            if (
                pre_step_at_sea.get(vessel.vessel_id, False)
                and not vessel.at_sea
                and vessel.location == vessel.destination
            ):
                log_event(
                    "vessel_arrived",
                    "phase3",
                    vessel_id=int(vessel.vessel_id),
                    port_id=int(vessel.location),
                    arrival_step=float(
                        vessel_step_stats.get(vessel.vessel_id, {}).get("arrival_step", self.t)
                    ),
                )
                if self._single_mission_enabled() and self._mission_success_on() == "arrival":
                    self._mark_vessel_mission_terminal(vessel, success=True)
                    step_mission_successes += 1.0
                    log_event(
                        "mission_succeeded",
                        "phase3",
                        vessel_id=int(vessel.vessel_id),
                        reason="arrival",
                    )
                else:
                    self._enqueue_arrived_vessel(vessel.location, vessel.vessel_id)
                    log_event(
                        "arrival_queued",
                        "phase3",
                        vessel_id=int(vessel.vessel_id),
                        port_id=int(vessel.location),
                        queue_length=float(self.ports[vessel.location].queue),
                    )
            deadline_step = float(pre_step_deadline.get(vessel.vessel_id, 0.0))
            arrived = bool(vessel_step_stats.get(vessel.vessel_id, {}).get("arrived", False))
            if deadline_step > 0.0 and (
                pre_step_trip_active.get(vessel.vessel_id, False)
                or vessel.at_sea
                or vessel.pending_departure
                or arrived
            ):
                end_step = (
                    float(vessel_step_stats.get(vessel.vessel_id, {}).get("arrival_step", self.t + 1.0))
                    if arrived
                    else float(self.t + 1.0)
                )
                schedule_delay = self._schedule_delay_delta_hours(
                    deadline_step,
                    float(self.t),
                    end_step,
                )
                if schedule_delay > 0.0:
                    vessel.schedule_delay_hours += schedule_delay
                    step_schedule_delay_by_vessel[vessel.vessel_id] += schedule_delay
            if arrived:
                vessel.completed_arrivals += 1
                vessel.pending_requested_arrival_time = 0.0
                if deadline_step > 0.0:
                    total_schedule_delay = self._schedule_delay_delta_hours(
                        deadline_step,
                        0.0,
                        float(vessel_step_stats.get(vessel.vessel_id, {}).get("arrival_step", self.t)),
                    )
                    vessel.completed_scheduled_arrivals += 1
                    step_scheduled_arrivals_completed += 1.0
                    vessel.last_schedule_delay_hours = total_schedule_delay
                    if total_schedule_delay <= float(self.cfg.get("on_time_tolerance_hours", 2.0)) + 1e-9:
                        vessel.on_time_arrivals += 1
                        step_on_time_arrivals += 1.0
                        step_on_time_arrival_by_vessel[vessel.vessel_id] = True
                else:
                    vessel.last_schedule_delay_hours = 0.0
                vessel.requested_arrival_time = 0.0

        # ── Phase 4: port action + service tick ──────────────────────────────
        # Ports (on their sub-cadence) accept/reject pending berth requests and
        # enqueue slot responses.  step_ports() then drains completed berths,
        # admits queued vessels, and accumulates per-port wait-time statistics.
        port_inputs = self._normalize_port_actions(actions)
        step_accepted_by_port = [0.0 for _ in self.ports]
        step_rejected_by_port = [0.0 for _ in self.ports]
        if due["port"]:
            normalized_port_actions: list[dict[str, Any]] = []
            for port_id, (port_agent, action) in enumerate(zip(self.port_agents, port_inputs)):
                normalized = port_agent.apply_action(action)
                normalized_port_actions.append(normalized)
                backlog = self.bus.get_pending_requests_sorted(port_id)
                accept_limit = min(
                    len(backlog),
                    max(int(normalized.get("accept_requests", 0)), 0),
                )
                accepted = backlog[:accept_limit]
                rejected = backlog[accept_limit:]
                step_accepted_by_port[port_id] = float(len(accepted))
                step_rejected_by_port[port_id] = float(len(rejected))
                step_requests_accepted += len(accepted)
                step_requests_rejected += len(rejected)
                self.bus.clear_pending_requests(port_id)
                response_step = self.t + self.cadence.message_latency_steps
                for vessel_id in accepted:
                    self.bus.enqueue_slot_response(response_step, vessel_id, True, port_id)
                    log_event(
                        "slot_response_enqueued",
                        "phase4",
                        vessel_id=int(vessel_id),
                        port_id=int(port_id),
                        accepted=True,
                        deliver_step=int(response_step),
                    )
                for vessel_id in rejected:
                    self.bus.enqueue_slot_response(response_step, vessel_id, False, port_id)
                    log_event(
                        "slot_response_enqueued",
                        "phase4",
                        vessel_id=int(vessel_id),
                        port_id=int(port_id),
                        accepted=False,
                        deliver_step=int(response_step),
                    )
            self._last_port_actions = normalized_port_actions

        service_rates = [
            int(action.get("service_rate", 1))
            for action in (
                self._last_port_actions
                if self._last_port_actions
                else [port_agent.last_action for port_agent in self.port_agents]
            )
        ]
        pre_step_port_served = [float(port.vessels_served) for port in self.ports]
        completed_service_by_port = step_ports(
            self.ports,
            service_rates,
            dt_hours=dt_hours,
            service_time_hours=self.cfg["service_time_hours"],
        )
        for port_id, completed_vessel_ids in enumerate(completed_service_by_port):
            for vessel_id in completed_vessel_ids:
                if 0 <= vessel_id < len(self.vessels):
                    vessel = self.vessels[vessel_id]
                    log_event(
                        "service_completed",
                        "phase4",
                        vessel_id=int(vessel_id),
                        port_id=int(port_id),
                    )
                    if self._single_mission_enabled() and self._mission_success_on() == "service_complete":
                        if not vessel.mission_done:
                            self._mark_vessel_mission_terminal(vessel, success=True)
                            step_mission_successes += 1.0
                            log_event(
                                "mission_succeeded",
                                "phase4",
                                vessel_id=int(vessel_id),
                                port_id=int(port_id),
                                reason="service_complete",
                            )
                    else:
                        vessel.fuel = float(vessel.initial_fuel)
                        vessel.stalled = False
                        step_refueled_vessels += 1.0
                        log_event(
                            "vessel_refueled",
                            "phase4",
                            vessel_id=int(vessel_id),
                            port_id=int(port_id),
                            fuel=float(vessel.fuel),
                        )
        self._sync_vessel_port_service_state()
        step_served_by_port = [
            float(port.vessels_served) - pre
            for port, pre in zip(self.ports, pre_step_port_served)
        ]
        step_vessels_served = float(sum(step_served_by_port))

        # ── Phase 5: reward computation ──────────────────────────────────────
        # Rewards are computed from the state produced by phases 1–4, using
        # weather W_t (the value active during this tick).  The clock and
        # weather are advanced AFTER rewards so that r_t = R(s_t, a_t, W_t).
        rewards = self._compute_rewards(
            vessel_step_stats=vessel_step_stats,
            step_delay_by_vessel=step_delay_by_vessel,
            step_schedule_delay_by_vessel=step_schedule_delay_by_vessel,
            step_on_time_arrival_by_vessel=step_on_time_arrival_by_vessel,
            step_served_by_port=step_served_by_port,
            step_vessels_served=step_vessels_served,
            step_accepted_by_port=step_accepted_by_port,
            step_rejected_by_port=step_rejected_by_port,
            step_requests_accepted=float(step_requests_accepted),
            step_requests_rejected=float(step_requests_rejected),
        )

        # ── Advance clock and environment stochastic state ───────────────────
        # t increments here; _refresh_weather() advances W_t → W_{t+1} via
        # AR(1).  Observations returned below therefore contain W_{t+1}, which
        # agents should treat as the weather forecast for their *next* action.
        terminated = bool(
            self._single_mission_enabled()
            and self.vessels
            and all(bool(v.mission_done) for v in self.vessels)
        )
        self.t += 1
        truncated = bool((not terminated) and self.t >= self.cfg["rollout_steps"])
        done = bool(terminated or truncated)
        if terminated:
            self._last_done_reason = "all_missions_complete"
        elif truncated:
            self._last_done_reason = "rollout_limit"
        else:
            self._last_done_reason = "in_progress"
        self._refresh_weather()
        self._refresh_forecasts()
        obs = self._get_observations()
        pending_requests, booked_arrivals, imminent_arrivals, reservation_pressure = self._port_load_snapshot()
        info = {
            "port_metrics": compute_port_metrics(self.ports),
            "cadence_due": due,
            "coordinator_assignments": assignments,
            "message_queues": self.bus.queue_sizes,
            "pending_arrival_requests": float(self.bus.total_pending_requests),
            "pending_requests_by_port": pending_requests.tolist(),
            "booked_arrivals_by_port": booked_arrivals.tolist(),
            "imminent_arrivals_by_port": imminent_arrivals.tolist(),
            "reservation_pressure_by_port": reservation_pressure.tolist(),
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
            "step_schedule_delay_hours": float(sum(step_schedule_delay_by_vessel.values())),
            "step_stall_hours": float(
                sum(float(stats.get("stall_hours", 0.0)) for stats in vessel_step_stats.values())
            ),
            "step_vessels_served": step_vessels_served,
            "step_scheduled_arrivals_completed": step_scheduled_arrivals_completed,
            "step_on_time_arrivals": step_on_time_arrivals,
            "step_fuel_capped_departures": step_fuel_capped_departures,
            "step_fuel_blocked_departures": step_fuel_blocked_departures,
            "step_refueled_vessels": step_refueled_vessels,
            "step_mission_successes": step_mission_successes,
            "step_mission_failures": step_mission_failures,
            "mission_successes": float(sum(bool(v.mission_success) for v in self.vessels)),
            "mission_failures": float(sum(bool(v.mission_failed) for v in self.vessels)),
            "mission_done": float(sum(bool(v.mission_done) for v in self.vessels)),
            "stalled_vessels": float(sum(bool(v.stalled) for v in self.vessels)),
            # Fleet utilization diagnostics
            "active_vessel_count": float(sum(
                bool(v.at_sea or v.pending_departure) for v in self.vessels
            )),
            "vessel_utilization_rate": (
                float(sum(
                    bool(v.at_sea or v.pending_departure or int(getattr(v, "port_service_state", 0)) > 0)
                    for v in self.vessels
                )) / max(len(self.vessels), 1)
            ),
            # Per-port wait diagnostics
            "mean_wait_by_port": [
                float(p.cumulative_wait_hours) / max(int(p.vessels_served), 1)
                for p in self.ports
            ],
            # On-time compliance rate
            "on_time_rate": (
                float(sum(v.on_time_arrivals for v in self.vessels))
                / max(float(sum(v.completed_scheduled_arrivals for v in self.vessels)), 1.0)
            ),
            "weather_enabled": self._weather_enabled,
            "events": step_events,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "done_reason": self._last_done_reason,
            **self._aggregate_reward_components(rewards),
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
                weather=self._weather if self._weather_enabled else None,
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
                weather_features=self._port_weather_features(i),
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
        source = str(self.cfg.get("forecast_source", "heuristic"))
        if source == "noiseless_current":
            self.medium_forecast, self.short_forecast = self.noiseless_forecaster.predict(self.ports)
            return
        if source == "ground_truth":
            self.medium_forecast, self.short_forecast = self.ground_truth_forecaster.predict(
                self.ports,
                self.vessels,
                current_step=self.t,
                weather=self._weather if self._weather_enabled else None,
            )
            return
        self.medium_forecast = self.medium_forecaster.predict(self.ports, self.rng)
        self.short_forecast = self.short_forecaster.predict(self.ports, self.rng)

    def _port_weather_features(self, port_id: int) -> np.ndarray | None:
        """Return compact inbound-weather features for a port, or None if disabled.

        Feature vector (length 3):
        1) mean inbound sea state
        2) max inbound sea state
        3) fraction of inbound routes at/above rough threshold
        """
        if not (self._weather_enabled and bool(self.cfg.get("port_weather_features", True))):
            return None
        if self._weather is None:
            return np.zeros(3, dtype=float)
        n = self._weather.shape[0]
        if not (0 <= port_id < n):
            return np.zeros(3, dtype=float)

        inbound = np.asarray(self._weather[:, port_id], dtype=float)
        if inbound.size > 1:
            inbound = np.delete(inbound, port_id)  # exclude self-route diagonal
        if inbound.size == 0:
            return np.zeros(3, dtype=float)

        sea_max = float(self.cfg.get("sea_state_max", 3.0))
        rough_threshold = 0.7 * sea_max
        mean_inbound = float(np.mean(inbound))
        max_inbound = float(np.max(inbound))
        rough_fraction = float(np.mean(inbound >= rough_threshold))
        return np.array([mean_inbound, max_inbound, rough_fraction], dtype=float)

    def _sync_vessel_port_service_state(self) -> None:
        """Mirror port queue/service membership onto vessel-local service state."""
        for vessel in self.vessels:
            vessel.port_service_state = 0
        for port in self.ports:
            for vessel_id in getattr(port, "queued_vessel_ids", []):
                if 0 <= vessel_id < len(self.vessels):
                    self.vessels[vessel_id].port_service_state = 1
            for vessel_id in getattr(port, "servicing_vessel_ids", []):
                if 0 <= vessel_id < len(self.vessels):
                    self.vessels[vessel_id].port_service_state = 2

    def _is_vessel_port_busy(self, vessel_id: int) -> bool:
        """Return True when a vessel is queued or in service at a port."""
        if not (0 <= vessel_id < len(self.vessels)):
            return False
        return int(getattr(self.vessels[vessel_id], "port_service_state", 0)) > 0

    def _enqueue_arrived_vessel(self, port_id: int, vessel_id: int) -> None:
        """Append a real vessel to a port queue without double-counting it."""
        if not (0 <= port_id < len(self.ports)):
            return
        port = self.ports[port_id]
        if vessel_id in getattr(port, "queued_vessel_ids", ()):
            port.queue = len(port.queued_vessel_ids)
            return
        if vessel_id in getattr(port, "servicing_vessel_ids", ()):
            port.occupied = len(port.service_times)
            return
        port.queued_vessel_ids.append(int(vessel_id))
        port.queue = len(port.queued_vessel_ids)

    def _estimate_arrival_hours(self, vessel: VesselState, destination: int) -> float:
        """Estimate hours until a vessel can reach *destination*.

        Returns ``np.inf`` when the vessel is stalled, stationary, or the
        destination is invalid. For ``pending_departure`` vessels, the wait
        until ``depart_at_step`` is included.
        """
        if not (0 <= destination < self.num_ports):
            return float(np.inf)
        if bool(getattr(vessel, "mission_done", False)):
            return float(np.inf)
        if bool(vessel.stalled) or float(vessel.speed) <= 0.0 or float(vessel.fuel) <= 0.0:
            return float(np.inf)

        dt_hours = float(self.cfg.get("dt_hours", 1.0))
        wait_hours = 0.0
        position_nm = float(vessel.position_nm)
        if vessel.pending_departure:
            wait_hours = max(int(vessel.depart_at_step) - int(self.t), 0) * dt_hours
            position_nm = 0.0
        elif not vessel.at_sea:
            return float(np.inf)

        sea_state = 0.0
        if self._weather_enabled and self._weather is not None:
            src = int(vessel.location)
            if 0 <= src < self._weather.shape[0] and 0 <= destination < self._weather.shape[1]:
                sea_state = float(self._weather[src, destination])
        effective_speed = float(vessel.speed) * weather_speed_factor(
            sea_state,
            float(self.cfg.get("weather_penalty_factor", 0.15)),
        )
        if effective_speed <= 0.0:
            return float(np.inf)
        leg_distance = float(self.distance_nm[int(vessel.location), destination])
        remaining_distance = max(leg_distance - position_nm, 0.0)
        return wait_hours + (remaining_distance / effective_speed if remaining_distance > 0.0 else 0.0)

    def _route_sea_state(self, origin: int, destination: int) -> float:
        """Return current sea state for a route, or 0.0 when unavailable."""
        if not self._weather_enabled or self._weather is None:
            return 0.0
        if not (0 <= origin < self._weather.shape[0] and 0 <= destination < self._weather.shape[1]):
            return 0.0
        return float(self._weather[origin, destination])

    def _route_fuel_requirement(
        self,
        vessel: VesselState,
        destination: int,
        speed: float,
    ) -> float:
        """Estimate total fuel needed to complete a fresh leg at *speed*."""
        if not (0 <= destination < self.num_ports):
            return float(np.inf)
        origin = int(vessel.location)
        leg_distance = float(self.distance_nm[origin, destination])
        if leg_distance <= 0.0:
            return 0.0
        sea_state = self._route_sea_state(origin, destination)
        effective_speed = float(speed) * weather_speed_factor(
            sea_state,
            float(self.cfg.get("weather_penalty_factor", 0.15)),
        )
        if effective_speed <= 0.0:
            return float(np.inf)
        travel_hours = leg_distance / effective_speed
        fuel_required, _ = compute_fuel_and_emissions(
            float(speed),
            self.cfg,
            hours=travel_hours,
            sea_state=sea_state,
        )
        return float(fuel_required)

    def _resolve_fuel_safe_dispatch_speed(
        self,
        vessel: VesselState,
        destination: int,
        requested_speed: float,
    ) -> tuple[float | None, bool]:
        """Return a safe dispatch speed, or ``None`` when departure is impossible.

        The returned tuple is ``(speed, capped)``, where ``capped=True`` means the
        requested speed would have exhausted fuel before arrival and was reduced to
        the fastest feasible speed for the full leg under current route conditions.
        """
        requested_speed = float(
            np.clip(requested_speed, self.cfg["speed_min"], self.cfg["speed_max"])
        )
        available_fuel = max(float(vessel.fuel), 0.0)
        if available_fuel <= 0.0:
            return None, False
        requested_fuel = self._route_fuel_requirement(vessel, destination, requested_speed)
        if requested_fuel <= available_fuel + 1e-9:
            return requested_speed, False

        min_speed = float(self.cfg["speed_min"])
        min_speed_fuel = self._route_fuel_requirement(vessel, destination, min_speed)
        if min_speed_fuel > available_fuel + 1e-9:
            return None, False

        lo = min_speed
        hi = requested_speed
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if self._route_fuel_requirement(vessel, destination, mid) <= available_fuel + 1e-9:
                lo = mid
            else:
                hi = mid
        return float(lo), True

    def _baseline_requested_arrival_time(
        self,
        vessel: VesselState,
        destination: int,
        departure_window_hours: int = 0,
    ) -> float:
        """Return a feasible default arrival target in absolute step units."""
        dt_hours = float(self.cfg.get("dt_hours", 1.0))
        if dt_hours <= 0.0:
            dt_hours = 1.0
        depart_steps = max(float(departure_window_hours), 0.0) / dt_hours
        nominal_speed = max(float(self.cfg.get("nominal_speed", vessel.speed)), 1e-6)
        if 0 <= destination < self.num_ports:
            leg_distance = float(self.distance_nm[int(vessel.location), destination])
        else:
            leg_distance = 0.0
        travel_hours = leg_distance / nominal_speed if leg_distance > 0.0 else dt_hours
        slack_hours = float(
            self.cfg.get(
                "requested_arrival_slack_hours",
                self.cfg.get("service_time_hours", dt_hours),
            )
        )
        travel_steps = max(travel_hours / dt_hours, 1.0)
        slack_steps = max(slack_hours / dt_hours, 0.0)
        return float(self.t) + depart_steps + travel_steps + slack_steps

    def _resolve_requested_arrival_time(
        self,
        vessel: VesselState,
        destination: int,
        requested_arrival_time: float,
        departure_window_hours: int = 0,
    ) -> float:
        """Resolve a vessel ETA request into a concrete absolute target step."""
        baseline = self._baseline_requested_arrival_time(
            vessel,
            destination,
            departure_window_hours=departure_window_hours,
        )
        if requested_arrival_time <= 0.0:
            return baseline
        return max(float(requested_arrival_time), float(self.t) + 1e-6)

    def _schedule_delay_delta_hours(
        self,
        deadline_step: float,
        start_step: float,
        end_step: float,
    ) -> float:
        """Return additional schedule lateness accrued between two step markers."""
        if deadline_step <= 0.0 or end_step <= start_step:
            return 0.0
        dt_hours = max(float(self.cfg.get("dt_hours", 1.0)), 1e-6)
        overdue_start = max(float(start_step) - float(deadline_step), 0.0)
        overdue_end = max(float(end_step) - float(deadline_step), 0.0)
        return max(overdue_end - overdue_start, 0.0) * dt_hours

    def _port_load_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-port pending, booked, imminent, and total reservation pressure.

        ``booked_arrivals`` counts vessels already committed to a port via
        ``at_sea`` / ``pending_departure`` state, plus accepted slot responses
        still queued in the message bus and not yet delivered to vessels.
        """
        pending_requests = np.array(
            [len(self.bus.get_pending_requests(port_id)) for port_id in range(self.num_ports)],
            dtype=float,
        )
        booked_arrivals = np.zeros(self.num_ports, dtype=float)
        imminent_arrivals = np.zeros(self.num_ports, dtype=float)
        imminent_horizon = float(self.cfg.get("short_horizon_hours", 0))
        for vessel in self.vessels:
            if bool(getattr(vessel, "mission_done", False)):
                continue
            if not (vessel.at_sea or vessel.pending_departure):
                continue
            dest = int(vessel.destination)
            if 0 <= dest < self.num_ports:
                booked_arrivals[dest] += 1.0
                eta_hours = self._estimate_arrival_hours(vessel, dest)
                if eta_hours <= imminent_horizon + 1e-9:
                    imminent_arrivals[dest] += 1.0
        for port_id, count in self.bus.count_slot_responses_by_port(accepted=True).items():
            if 0 <= port_id < self.num_ports:
                booked_arrivals[port_id] += float(count)

        queue_load = np.array([float(port.queue) for port in self.ports], dtype=float)
        reservation_pressure = queue_load + pending_requests + booked_arrivals
        return pending_requests, booked_arrivals, imminent_arrivals, reservation_pressure

    def _get_observations(self) -> dict[str, Any]:
        """Build observation dicts for all agent types from current state.

        Coordinator observations are **zero-padded** to the maximum
        coordinator dimension
        (``num_ports * medium_d + num_ports * 5 + num_vessels * 7 + 1``
        ``[+ num_ports*num_ports if weather]``) so that parameter-shared
        networks receive fixed-size inputs even when vessel partitions vary.
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
        weather_dim = (
            int(self.cfg["num_ports"]) * int(self.cfg["num_ports"])
            if bool(self.cfg.get("weather_enabled", True))
            else 0
        )
        pending_requests, booked_arrivals, imminent_arrivals, reservation_pressure = self._port_load_snapshot()
        port_load_summary = np.column_stack(
            [
                np.array([float(port.queue) for port in self.ports], dtype=float),
                np.array([float(port.occupied) for port in self.ports], dtype=float),
                np.array([float(port.docks) for port in self.ports], dtype=float),
                booked_arrivals,
                imminent_arrivals,
            ]
        ).flatten()
        max_coord_dim = (
            int(self.cfg["num_ports"]) * medium_d
            + int(self.cfg["num_ports"]) * 5
            + self.num_vessels * 7
            + 1
            + weather_dim
        )

        coordinator_obs = []
        for coordinator_id, coordinator in enumerate(self.coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in self.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = self.vessels
            raw_obs = coordinator.get_obs(
                medium,
                local_vessels,
                port_load_summary=port_load_summary,
                weather=self._weather if self._weather_enabled else None,
            )
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
            # Remaining range to destination (nm)
            if vessel.at_sea and 0 <= int(vessel.location) < self.num_ports:
                leg_dist = float(self.distance_nm[int(vessel.location), dest])
                remaining_range_nm = max(leg_dist - float(vessel.position_nm), 0.0)
            else:
                remaining_range_nm = 0.0
            # Hours until requested arrival deadline (0.0 when no deadline)
            dt_hours = max(float(self.cfg.get("dt_hours", 1.0)), 1e-6)
            if float(vessel.requested_arrival_time) > 0.0:
                deadline_delta_hours = max(
                    (float(vessel.requested_arrival_time) - float(self.t)) * dt_hours,
                    0.0,
                )
            else:
                deadline_delta_hours = 0.0
            v_obs = vessel_agent.get_obs(
                short[dest],
                directive=directive,
                dock_availability=dock_avail,
                remaining_range_nm=remaining_range_nm,
                deadline_delta_hours=deadline_delta_hours,
            )
            # Append weather sea state for the vessel's current route
            if self._weather_enabled and self._weather is not None:
                src = vessel.location
                sea_state = float(self._weather[src, dest])
                v_obs = np.concatenate([v_obs, np.array([sea_state])])
            vessel_obs.append(v_obs)

        port_obs = []
        for i, port_agent in enumerate(self.port_agents):
            weather_features = self._port_weather_features(i)
            p_obs = port_agent.get_obs(
                short[i],
                incoming_requests=int(pending_requests[i]),
                booked_arrivals=float(booked_arrivals[i]),
                imminent_arrivals=float(imminent_arrivals[i]),
                weather_features=weather_features,
            )
            port_obs.append(p_obs)

        return {
            "coordinator": self._sanitize_obs(coord_obs),
            "coordinators": [self._sanitize_obs(o) for o in coordinator_obs],
            "vessels": [self._sanitize_obs(o) for o in vessel_obs],
            "ports": [self._sanitize_obs(o) for o in port_obs],
        }

    @staticmethod
    def _sanitize_obs(obs: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with zero to prevent neural-network corruption."""
        if not np.all(np.isfinite(obs)):
            return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _compute_rewards(
        self,
        vessel_step_stats: dict[int, dict[str, float | bool]] | None = None,
        step_delay_by_vessel: dict[int, float] | None = None,
        step_schedule_delay_by_vessel: dict[int, float] | None = None,
        step_on_time_arrival_by_vessel: dict[int, bool] | None = None,
        step_served_by_port: list[float] | None = None,
        step_vessels_served: float = 0.0,
        step_accepted_by_port: list[float] | None = None,
        step_rejected_by_port: list[float] | None = None,
        step_requests_accepted: float = 0.0,
        step_requests_rejected: float = 0.0,
    ) -> dict[str, Any]:
        """Compute per-agent rewards from step-level physics deltas.

        Called in Phase 5 of the transition kernel, **before** the clock
        advances and weather updates.  Therefore rewards use ``W_t`` — the
        weather that was active during phases 1–4.

        Returns
        -------
        dict with keys:
            ``"vessels"``       — list[float], one reward per vessel.
            ``"ports"``         — list[float], one reward per port.
            ``"coordinator"``   — float, first coordinator's reward.
            ``"coordinators"``  — list[float], one per coordinator.

        Reward composition per agent type:
            * Vessel:      base step reward + weather vessel shaping (if enabled).
            * Port:        base step reward (queue wait + idle dock penalty).
            * Coordinator: base step reward + weather coordinator shaping (if enabled).
        """
        vessel_step_stats = vessel_step_stats or {}
        step_delay_by_vessel = step_delay_by_vessel or {}
        step_schedule_delay_by_vessel = step_schedule_delay_by_vessel or {}
        step_on_time_arrival_by_vessel = step_on_time_arrival_by_vessel or {}
        step_served_by_port = step_served_by_port or [0.0 for _ in self.ports]
        step_accepted_by_port = step_accepted_by_port or [0.0 for _ in self.ports]
        step_rejected_by_port = step_rejected_by_port or [0.0 for _ in self.ports]
        vessel_components = [
            compute_vessel_reward_breakdown(
                vessel=v,
                config=self.cfg,
                fuel_used=float(vessel_step_stats.get(v.vessel_id, {}).get("fuel_used", 0.0)),
                co2_emitted=float(vessel_step_stats.get(v.vessel_id, {}).get("co2_emitted", 0.0)),
                delay_hours=float(step_delay_by_vessel.get(v.vessel_id, 0.0)),
                schedule_delay_hours=float(
                    step_schedule_delay_by_vessel.get(v.vessel_id, 0.0)
                ),
                transit_hours=float(
                    vessel_step_stats.get(v.vessel_id, {}).get("travel_hours", 0.0)
                ),
                arrived=bool(vessel_step_stats.get(v.vessel_id, {}).get("arrived", False)),
                arrived_on_time=bool(step_on_time_arrival_by_vessel.get(v.vessel_id, False)),
            )
            for v in self.vessels
        ]
        vessel_rewards = [float(parts["total"]) for parts in vessel_components]
        # Apply weather shaping bonus for vessels that slow in rough seas.
        # Only at-sea vessels receive shaping — docked vessels make no routing decisions.
        if self._weather_enabled and self._weather is not None:
            for i, v in enumerate(self.vessels):
                if not v.at_sea or bool(v.stalled):
                    continue
                speed = float(v.speed)
                src, dst = v.location, v.destination
                n = self._weather.shape[0]
                sea = float(self._weather[src, dst]) if 0 <= src < n and 0 <= dst < n else 0.0
                bonus = float(weather_vessel_shaping(speed, sea, self.cfg))
                vessel_components[i]["weather_shaping_bonus"] += bonus
                vessel_components[i]["total"] += bonus
                vessel_rewards[i] += bonus
        port_components = [
            compute_port_reward_breakdown(
                p,
                self.cfg,
                served_vessels=step_served_by_port[idx],
                accepted_requests=step_accepted_by_port[idx],
                rejected_requests=step_rejected_by_port[idx],
            )
            for idx, p in enumerate(self.ports)
        ]
        port_rewards = [float(parts["total"]) for parts in port_components]
        step_fuel_used = float(
            sum(float(stats.get("fuel_used", 0.0)) for stats in vessel_step_stats.values())
        )
        step_co2_emitted = float(
            sum(float(stats.get("co2_emitted", 0.0)) for stats in vessel_step_stats.values())
        )
        # Directive compliance: fraction of vessels heading to assigned dest.
        # Each vessel's directive contains a per-vessel "dest_port" (set in
        # phase 1 from per_vessel_dest or the coordinator's primary dest).
        compliance_count = 0
        compliance_total = 0
        for v in self.vessels:
            d = self.bus.get_latest_directive(v.vessel_id)
            if d and "dest_port" in d:
                assigned = d["dest_port"]
                compliance_total += 1
                if v.destination == int(assigned):
                    compliance_count += 1
        compliance_rate = float(compliance_count) / max(compliance_total, 1)
        coordinator_components = [
            compute_coordinator_reward_breakdown(
                ports=self.ports,
                config=self.cfg,
                fuel_used=step_fuel_used,
                co2_emitted=step_co2_emitted,
                delay_hours=float(sum(step_delay_by_vessel.values())),
                schedule_delay_hours=float(sum(step_schedule_delay_by_vessel.values())),
                served_vessels=float(step_vessels_served),
                accepted_requests=float(step_requests_accepted),
                rejected_requests=float(step_requests_rejected),
                directive_compliance_rate=compliance_rate,
            )
            for _ in self.coordinators
        ]
        coordinator_rewards = [float(parts["total"]) for parts in coordinator_components]
        # Apply weather shaping bonus for coordinators routing through calm seas.
        if self._weather_enabled and self._weather is not None:
            routes: list[tuple[int, int]] = []
            for v in self.vessels:
                d = self.bus.get_latest_directive(v.vessel_id)
                if d:
                    routes.append((v.location, int(d.get("dest_port", v.destination))))
            if routes:
                bonus = float(weather_coordinator_shaping(self._weather, routes, self.cfg))
                for i in range(len(coordinator_rewards)):
                    coordinator_components[i]["weather_shaping_bonus"] += bonus
                    coordinator_components[i]["total"] += bonus
                    coordinator_rewards[i] += bonus
        return {
            "coordinator": coordinator_rewards[0],
            "coordinators": coordinator_rewards,
            "vessels": vessel_rewards,
            "ports": port_rewards,
            "coordinator_components": coordinator_components,
            "vessel_components": vessel_components,
            "port_components": port_components,
        }

    @staticmethod
    def _aggregate_reward_components(rewards: dict[str, Any]) -> dict[str, float]:
        """Aggregate structured reward breakdowns into flat step-level scalars."""
        flat: dict[str, float] = {}
        for prefix, parts in (
            ("vessel_reward", rewards.get("vessel_components", [])),
            ("port_reward", rewards.get("port_components", [])),
            ("coordinator_reward", rewards.get("coordinator_components", [])),
        ):
            if not parts:
                continue
            keys = sorted({str(k) for comp in parts for k in comp.keys()})
            for key in keys:
                column = f"{prefix}_total" if key == "total" else f"{prefix}_{key}_total"
                flat[column] = float(
                    sum(float(comp.get(key, 0.0)) for comp in parts)
                )
        return flat

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
