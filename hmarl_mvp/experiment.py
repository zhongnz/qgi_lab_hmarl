"""Experiment runner and baseline/ablation utilities."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import SEED, get_default_config
from .env import MaritimeEnv
from .forecasts import (
    GroundTruthForecaster,
    MediumTermForecaster,
    NoiselessForecaster,
    ShortTermForecaster,
)
from .learned_forecaster import LearnedForecaster
from .metrics import (
    compute_economic_metrics,
    compute_economic_step_deltas,
    compute_port_metrics,
    compute_vessel_metrics,
)
from .policies import (
    FleetCoordinatorPolicy,
    PortPolicy,
    VesselPolicy,
)
from .state import make_rng

VALID_POLICIES = {
    "independent",
    "reactive",
    "forecast",
    "noiseless",
    "ground_truth",
    "learned_forecast",
}


def _json_cell(value: Any) -> Any:
    """Serialize structured action/event payloads for CSV-friendly logging."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _flatten_vessel_state(vessels: list[Any], rewards: list[Any] | None = None) -> dict[str, float]:
    """Flatten per-vessel state into numeric columns for diagnostics."""
    flat: dict[str, float] = {}
    reward_list = list(rewards) if rewards is not None else []
    for vessel in vessels:
        prefix = f"vessel_{int(vessel.vessel_id)}"
        flat[f"{prefix}_location"] = float(vessel.location)
        flat[f"{prefix}_destination"] = float(vessel.destination)
        flat[f"{prefix}_position_nm"] = float(vessel.position_nm)
        flat[f"{prefix}_speed"] = float(vessel.speed)
        flat[f"{prefix}_fuel"] = float(vessel.fuel)
        flat[f"{prefix}_cumulative_fuel_used"] = float(
            getattr(vessel, "cumulative_fuel_used", 0.0)
        )
        flat[f"{prefix}_emissions"] = float(vessel.emissions)
        flat[f"{prefix}_delay_hours"] = float(vessel.delay_hours)
        flat[f"{prefix}_schedule_delay_hours"] = float(
            getattr(vessel, "schedule_delay_hours", 0.0)
        )
        flat[f"{prefix}_at_sea"] = float(bool(vessel.at_sea))
        flat[f"{prefix}_stalled"] = float(bool(getattr(vessel, "stalled", False)))
        flat[f"{prefix}_port_service_state"] = float(
            getattr(vessel, "port_service_state", 0)
        )
        flat[f"{prefix}_requested_arrival_time"] = float(
            getattr(vessel, "requested_arrival_time", 0.0)
        )
        flat[f"{prefix}_pending_requested_arrival_time"] = float(
            getattr(vessel, "pending_requested_arrival_time", 0.0)
        )
        flat[f"{prefix}_completed_arrivals"] = float(getattr(vessel, "completed_arrivals", 0))
        flat[f"{prefix}_completed_scheduled_arrivals"] = float(
            getattr(vessel, "completed_scheduled_arrivals", 0)
        )
        flat[f"{prefix}_on_time_arrivals"] = float(getattr(vessel, "on_time_arrivals", 0))
        flat[f"{prefix}_last_schedule_delay_hours"] = float(
            getattr(vessel, "last_schedule_delay_hours", 0.0)
        )
        flat[f"{prefix}_mission_done"] = float(bool(getattr(vessel, "mission_done", False)))
        flat[f"{prefix}_mission_success"] = float(
            bool(getattr(vessel, "mission_success", False))
        )
        flat[f"{prefix}_mission_failed"] = float(bool(getattr(vessel, "mission_failed", False)))
        flat[f"{prefix}_pending_departure"] = float(bool(vessel.pending_departure))
        flat[f"{prefix}_depart_at_step"] = float(vessel.depart_at_step)
        vid = int(vessel.vessel_id)
        if 0 <= vid < len(reward_list):
            flat[f"{prefix}_reward"] = float(reward_list[vid])
    return flat


def _flatten_port_state(
    ports: list[Any],
    rewards: list[Any] | None = None,
    pending_requests: list[float] | None = None,
    booked_arrivals: list[float] | None = None,
    imminent_arrivals: list[float] | None = None,
    reservation_pressure: list[float] | None = None,
) -> dict[str, float]:
    """Flatten per-port state into numeric columns for diagnostics."""
    flat: dict[str, float] = {}
    reward_list = list(rewards) if rewards is not None else []
    pending_list = list(pending_requests) if pending_requests is not None else []
    booked_list = list(booked_arrivals) if booked_arrivals is not None else []
    imminent_list = list(imminent_arrivals) if imminent_arrivals is not None else []
    pressure_list = list(reservation_pressure) if reservation_pressure is not None else []
    for port in ports:
        prefix = f"port_{int(port.port_id)}"
        flat[f"{prefix}_queue"] = float(port.queue)
        flat[f"{prefix}_docks"] = float(port.docks)
        flat[f"{prefix}_occupied"] = float(port.occupied)
        flat[f"{prefix}_available_docks"] = float(max(port.docks - port.occupied, 0))
        flat[f"{prefix}_utilization"] = float(port.occupied / max(port.docks, 1))
        flat[f"{prefix}_service_count"] = float(len(port.service_times))
        flat[f"{prefix}_cumulative_wait_hours"] = float(port.cumulative_wait_hours)
        flat[f"{prefix}_vessels_served"] = float(port.vessels_served)
        pid = int(port.port_id)
        if 0 <= pid < len(pending_list):
            flat[f"{prefix}_pending_requests"] = float(pending_list[pid])
        if 0 <= pid < len(booked_list):
            flat[f"{prefix}_booked_arrivals"] = float(booked_list[pid])
        if 0 <= pid < len(imminent_list):
            flat[f"{prefix}_imminent_arrivals"] = float(imminent_list[pid])
        if 0 <= pid < len(pressure_list):
            flat[f"{prefix}_reservation_pressure"] = float(pressure_list[pid])
        if 0 <= pid < len(reward_list):
            flat[f"{prefix}_reward"] = float(reward_list[pid])
    return flat


def _flatten_coordinator_rewards(rewards: list[Any] | None = None) -> dict[str, float]:
    """Flatten per-coordinator rewards into numeric columns for diagnostics."""
    flat: dict[str, float] = {}
    reward_list = list(rewards) if rewards is not None else []
    for idx, reward in enumerate(reward_list):
        flat[f"coordinator_{idx}_reward"] = float(reward)
    return flat


def _build_step_trace_row(
    *,
    t: int,
    policy: str,
    rewards: dict[str, Any],
    info: dict[str, Any],
    vessels: list[Any],
    ports: list[Any],
    config: dict[str, Any],
    num_coordinators: int,
    cumulative_vessel_requests: float,
    cumulative_port_accepted: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one per-step diagnostics row shared by heuristic and MAPPO traces."""
    vessel_metrics = compute_vessel_metrics(vessels)
    port_metrics = compute_port_metrics(ports)
    economic_metrics = compute_economic_metrics(vessels, config)
    step_economic = compute_economic_step_deltas(
        step_fuel_used=float(info.get("step_fuel_used", 0.0)),
        step_co2_emitted=float(info.get("step_co2_emitted", 0.0)),
        step_delay_hours=float(info.get("step_delay_hours", 0.0)),
        config=config,
    )
    queue_sizes = info.get("message_queues", {}) or {}
    step_requests = float(info.get("requests_submitted", 0.0))
    step_accepted = float(info.get("requests_accepted", 0.0))
    step_rejected = float(info.get("requests_rejected", 0.0))
    pending_by_port = [float(x) for x in (info.get("pending_requests_by_port") or [])]
    booked_by_port = [float(x) for x in (info.get("booked_arrivals_by_port") or [])]
    imminent_by_port = [float(x) for x in (info.get("imminent_arrivals_by_port") or [])]
    pressure_by_port = [float(x) for x in (info.get("reservation_pressure_by_port") or [])]
    row: dict[str, Any] = {
        "t": t,
        "policy": policy,
        "num_coordinators": num_coordinators,
        "coordinator_updates": int(bool((info.get("cadence_due") or {}).get("coordinator", False))),
        "pending_arrival_requests": float(info.get("pending_arrival_requests", 0.0)),
        "total_booked_arrivals": float(sum(booked_by_port)),
        "total_imminent_arrivals": float(sum(imminent_by_port)),
        "total_reservation_pressure": float(sum(pressure_by_port)),
        "weather_enabled": int(bool(info.get("weather_enabled", False))),
        "mean_sea_state": float(info.get("mean_sea_state", 0.0)),
        "max_sea_state": float(info.get("max_sea_state", 0.0)),
        "directive_queue_size": float(queue_sizes.get("directives", 0.0)),
        "arrival_request_queue_size": float(queue_sizes.get("arrival_requests", 0.0)),
        "slot_response_queue_size": float(queue_sizes.get("slot_responses", 0.0)),
        "step_vessel_requests": step_requests,
        "step_port_accepted": step_accepted,
        "step_port_rejected": step_rejected,
        "total_vessel_requests": cumulative_vessel_requests,
        "total_port_accepted": cumulative_port_accepted,
        "policy_agreement_rate": (
            cumulative_port_accepted / cumulative_vessel_requests
            if cumulative_vessel_requests > 0
            else 0.0
        ),
        "step_fuel_used": float(info.get("step_fuel_used", 0.0)),
        "step_co2_emitted": float(info.get("step_co2_emitted", 0.0)),
        "step_delay_hours": float(info.get("step_delay_hours", 0.0)),
        "step_schedule_delay_hours": float(info.get("step_schedule_delay_hours", 0.0)),
        "step_stall_hours": float(info.get("step_stall_hours", 0.0)),
        "step_vessels_served": float(info.get("step_vessels_served", 0.0)),
        "step_scheduled_arrivals_completed": float(
            info.get("step_scheduled_arrivals_completed", 0.0)
        ),
        "step_on_time_arrivals": float(info.get("step_on_time_arrivals", 0.0)),
        "step_fuel_capped_departures": float(info.get("step_fuel_capped_departures", 0.0)),
        "step_fuel_blocked_departures": float(info.get("step_fuel_blocked_departures", 0.0)),
        "step_refueled_vessels": float(info.get("step_refueled_vessels", 0.0)),
        "step_mission_successes": float(info.get("step_mission_successes", 0.0)),
        "step_mission_failures": float(info.get("step_mission_failures", 0.0)),
        "done_reason": str(info.get("done_reason", "in_progress")),
        "terminated": float(info.get("terminated", 0.0)),
        "truncated": float(info.get("truncated", 0.0)),
        **vessel_metrics,
        **port_metrics,
        **economic_metrics,
        **step_economic,
        "avg_vessel_reward": float(np.mean(rewards["vessels"])) if rewards["vessels"] else 0.0,
        "avg_port_reward": float(np.mean(rewards["ports"])) if rewards["ports"] else 0.0,
        "coordinator_reward": float(rewards["coordinator"]),
        **_flatten_vessel_state(vessels, rewards=rewards.get("vessels")),
        **_flatten_port_state(
            ports,
            rewards=rewards.get("ports"),
            pending_requests=pending_by_port,
            booked_arrivals=booked_by_port,
            imminent_arrivals=imminent_by_port,
            reservation_pressure=pressure_by_port,
        ),
        **_flatten_coordinator_rewards(rewards.get("coordinators")),
    }
    row.update(
        {
            key: float(value)
            for key, value in info.items()
            if key.startswith(("vessel_reward_", "port_reward_", "coordinator_reward_"))
            and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    if extra:
        row.update(extra)
    return row


def _build_action_trace_rows(
    *,
    t: int,
    policy: str,
    coordinator_actions: list[dict[str, Any]],
    vessel_actions: list[dict[str, Any]],
    port_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build flat per-agent action rows for one environment step."""
    rows: list[dict[str, Any]] = []
    for agent_id, action in enumerate(vessel_actions):
        rows.append(
            {
                "t": int(t),
                "policy": policy,
                "agent_type": "vessel",
                "agent_id": int(agent_id),
                "target_speed": float(action.get("target_speed", 0.0)),
                "request_arrival_slot": int(bool(action.get("request_arrival_slot", False))),
                "requested_arrival_time": float(action.get("requested_arrival_time", 0.0)),
            }
        )
    for agent_id, action in enumerate(port_actions):
        rows.append(
            {
                "t": int(t),
                "policy": policy,
                "agent_type": "port",
                "agent_id": int(agent_id),
                "service_rate": int(action.get("service_rate", 0)),
                "accept_requests": int(action.get("accept_requests", 0)),
            }
        )
    for agent_id, action in enumerate(coordinator_actions):
        rows.append(
            {
                "t": int(t),
                "policy": policy,
                "agent_type": "coordinator",
                "agent_id": int(agent_id),
                "dest_port": int(action.get("dest_port", -1)),
                "departure_window_hours": int(action.get("departure_window_hours", 0)),
                "emission_budget": float(action.get("emission_budget", 0.0)),
                "assigned_vessel_ids_json": _json_cell(action.get("assigned_vessel_ids", [])),
                "per_vessel_dest_json": _json_cell(action.get("per_vessel_dest", {})),
            }
        )
    return rows


def _build_event_log_rows(
    *,
    default_t: int,
    policy: str,
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build flat event rows from the environment step info payload."""
    rows: list[dict[str, Any]] = []
    for raw_event in info.get("events", []) or []:
        row: dict[str, Any] = {"policy": policy}
        for key, value in raw_event.items():
            row[key] = _json_cell(value)
        row.setdefault("t", int(default_t))
        rows.append(row)
    return rows


def run_experiment(
    policy_type: str = "forecast",
    forecast_horizon: int = 12,
    forecast_noise: float = 0.5,
    share_forecasts: bool = True,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    distance_nm: np.ndarray | None = None,
    learned_forecaster: LearnedForecaster | None = None,
) -> pd.DataFrame:
    """
    Run one episode and return per-step metrics.

    policy_type:
    - independent: no coordination, no forecast usage
    - reactive: coordination using current state only (no forecast)
    - forecast: forecast-informed coordination with noisy forecasts
    - noiseless: forecast-informed coordination with current queue repeated forward
    - ground_truth: forecast-informed coordination with deterministic committed-state rollout
    - learned_forecast: uses a trained ``LearnedForecaster`` for predictions
    """
    if policy_type not in VALID_POLICIES:
        raise ValueError(f"Unknown policy_type={policy_type!r}. Expected one of {VALID_POLICIES}")
    if policy_type == "learned_forecast" and learned_forecaster is None:
        raise ValueError("learned_forecaster must be provided when policy_type='learned_forecast'")

    # For learned_forecast, use the same heuristic policy logic but with learned predictions
    effective_mode = (
        "forecast" if policy_type in {"learned_forecast", "ground_truth"} else policy_type
    )

    cfg = get_default_config(**(config or {}))
    steps = cfg["rollout_steps"] if steps is None else steps
    cfg["rollout_steps"] = int(steps)
    rng = make_rng(seed)
    env = MaritimeEnv(config=cfg, seed=seed, distance_nm=distance_nm)
    env.reset()

    num_coordinators = env.num_coordinators
    coordinator_policy = FleetCoordinatorPolicy(config=cfg, mode=effective_mode)
    vessel_policy = VesselPolicy(config=cfg, mode=effective_mode)
    port_policy = PortPolicy(config=cfg, mode=effective_mode)
    medium_forecaster = MediumTermForecaster(cfg["medium_horizon_days"])
    short_forecaster = ShortTermForecaster(forecast_horizon)
    noiseless_forecaster = NoiselessForecaster(
        medium_horizon_days=cfg["medium_horizon_days"],
        short_horizon_hours=forecast_horizon,
    )
    ground_truth_forecaster = GroundTruthForecaster(
        medium_horizon_days=cfg["medium_horizon_days"],
        short_horizon_hours=forecast_horizon,
        config=cfg,
        distance_nm=env.distance_nm,
    )

    log: list[dict[str, Any]] = []
    cumulative_vessel_requests = 0.0
    cumulative_port_accepted = 0.0

    for _ in range(steps):
        t = env.t
        step_context = env.peek_step_context()
        if policy_type == "noiseless":
            medium, short = noiseless_forecaster.predict(env.ports)
        elif policy_type == "ground_truth":
            medium, short = ground_truth_forecaster.predict(
                env.ports,
                env.vessels,
                current_step=t,
                weather=getattr(env, "_weather", None) if cfg.get("weather_enabled") else None,
            )
        elif policy_type == "learned_forecast" and learned_forecaster is not None:
            full_pred = learned_forecaster.predict(env.ports, rng=rng)
            medium_h = int(cfg["medium_horizon_days"])
            short_h = int(forecast_horizon)
            if full_pred.shape[1] >= medium_h:
                medium = full_pred[:, :medium_h]
            else:
                medium = np.pad(
                    full_pred, ((0, 0), (0, medium_h - full_pred.shape[1])), mode="edge"
                )
            if full_pred.shape[1] >= short_h:
                short = full_pred[:, :short_h]
            else:
                short = np.pad(
                    full_pred, ((0, 0), (0, short_h - full_pred.shape[1])), mode="edge"
                )
        else:
            medium = medium_forecaster.predict(ports=env.ports, rng=rng)
            short = short_forecaster.predict(ports=env.ports, rng=rng)
            if policy_type == "forecast" and forecast_noise > 0:
                medium += rng.normal(0, forecast_noise, medium.shape)
                short += rng.normal(0, forecast_noise, short.shape)
                medium = np.clip(medium, 0, None)
                short = np.clip(short, 0, None)

        assignments = step_context["assignments"]
        latest_directive_by_vessel = step_context["latest_directive_by_vessel"]
        pending_port_requests = step_context["pending_port_requests"]

        coordinator_actions: list[dict[str, Any]] = []
        for coordinator_id in range(num_coordinators):
            local_ids = assignments.get(coordinator_id, [])
            vessels_by_id = {v.vessel_id: v for v in env.vessels}
            local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
            if not local_vessels:
                local_vessels = env.vessels
            coordinator_actions.append(
                coordinator_policy.propose_action(
                    medium_forecast=medium,
                    vessels=local_vessels,
                    ports=env.ports,
                    rng=rng,
                    weather=getattr(env, "_weather", None),
                )
            )

        forecast_for_vessels = short if share_forecasts else np.zeros_like(short)
        vessel_actions: list[dict[str, Any]] = []
        weather_matrix = getattr(env, "_weather", None)
        for vessel_agent in env.vessel_agents:
            vessel = vessel_agent.state
            # Extract local sea-state for the vessel's current route segment.
            local_sea_state = 0.0
            if weather_matrix is not None:
                src = vessel.location
                dest_id = latest_directive_by_vessel.get(vessel.vessel_id, {}).get(
                    "dest_port", vessel.location
                )
                n_ports = weather_matrix.shape[0]
                if 0 <= src < n_ports and 0 <= dest_id < n_ports:
                    local_sea_state = float(weather_matrix[src, dest_id])
            directive = env.get_directive_for_vessel(
                vessel_id=vessel.vessel_id,
                assignments=assignments,
                latest_directive_by_vessel=latest_directive_by_vessel,
            )
            vessel_actions.append(
                vessel_policy.propose_action(
                    short_forecast=forecast_for_vessels,
                    directive=directive,
                    sea_state=local_sea_state,
                )
            )

        forecast_for_ports = short if share_forecasts else np.zeros_like(short)
        port_actions = [
            port_policy.propose_action(
                port_state=port_agent.state,
                incoming_requests=len(pending_port_requests.get(i, [])),
                short_forecast_row=forecast_for_ports[i],
                weather_features=env._port_weather_features(i),
            )
            for i, port_agent in enumerate(env.port_agents)
        ]
        actions = {
            "coordinator": coordinator_actions[0] if coordinator_actions else {},
            "coordinators": coordinator_actions,
            "vessels": vessel_actions,
            "ports": port_actions,
        }
        _, rewards, done, info = env.step(actions)

        step_requests = float(info.get("requests_submitted", 0.0))
        step_accepted = float(info.get("requests_accepted", 0.0))
        cumulative_vessel_requests += step_requests
        cumulative_port_accepted += step_accepted

        log.append(
            _build_step_trace_row(
                t=t,
                policy=policy_type,
                rewards=rewards,
                info=info,
                vessels=env.vessels,
                ports=env.ports,
                config=cfg,
                num_coordinators=num_coordinators,
                cumulative_vessel_requests=cumulative_vessel_requests,
                cumulative_port_accepted=cumulative_port_accepted,
                extra={
                    "forecast_horizon": forecast_horizon,
                    "forecast_noise": forecast_noise,
                    "share_forecasts": int(bool(share_forecasts)),
                },
            )
        )
        if done:
            break

    return pd.DataFrame(log)


def run_trained_mappo_trace(
    trainer: Any,
    num_steps: int | None = None,
    deterministic: bool = True,
    policy_label: str = "mappo",
    return_logs: bool = False,
    heuristic_coordinator: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a trained MAPPO policy for one episode and return per-step diagnostics."""
    import torch

    from .mappo import _nn_to_coordinator_action, _nn_to_port_action, _nn_to_vessel_action

    num_steps = int(trainer.cfg["rollout_steps"]) if num_steps is None else int(num_steps)
    obs = trainer.env.reset(seed=trainer._seed)

    for actor_critic in trainer.actor_critics.values():
        actor_critic.eval()

    device = trainer.device
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    cumulative_vessel_requests = 0.0
    cumulative_port_accepted = 0.0

    for step_i in range(num_steps):
        global_state = trainer.env.get_global_state()
        gs_tensor = torch.as_tensor(
            global_state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            vessel_actions = []
            for i, v_obs in enumerate(obs["vessels"]):
                v_obs_n = trainer._eval_normalize_obs(v_obs, "vessel")
                v_t = torch.as_tensor(
                    v_obs_n, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_t, _, _, _ = trainer._get_ac("vessel", i).get_action_and_value(
                    v_t, gs_tensor, deterministic=deterministic
                )
                speed_cap = trainer._vessel_weather_speed_cap(i)
                vessel_actions.append(
                    _nn_to_vessel_action(
                        action_t.squeeze(0),
                        trainer.cfg,
                        speed_cap=speed_cap,
                        current_step=trainer.env.t,
                    )
                )

            port_actions = []
            for i, p_obs in enumerate(obs["ports"]):
                p_obs_n = trainer._eval_normalize_obs(p_obs, "port")
                p_t = torch.as_tensor(
                    p_obs_n, dtype=torch.float32, device=device
                ).unsqueeze(0)
                p_mask = trainer._port_mask_tensor(i)
                action_t, _, _, _ = trainer._get_ac("port", i).get_action_and_value(
                    p_t, gs_tensor, deterministic=deterministic, action_mask=p_mask
                )
                port_actions.append(_nn_to_port_action(action_t.squeeze(0), i, trainer.env))

            if heuristic_coordinator:
                heuristic_actions = trainer.env.sample_stub_actions()
                coord_actions = list(heuristic_actions.get("coordinators", []))
            else:
                assignments = trainer.env._build_assignments()
                coord_actions = []
                c_mask = trainer._coordinator_mask_tensor()
                for i, c_obs in enumerate(obs["coordinators"]):
                    c_obs_n = trainer._eval_normalize_obs(c_obs, "coordinator")
                    c_t = torch.as_tensor(
                        c_obs_n, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action_t, _, _, _ = trainer._get_ac("coordinator", i).get_action_and_value(
                        c_t, gs_tensor, deterministic=deterministic, action_mask=c_mask
                    )
                    coord_actions.append(
                        _nn_to_coordinator_action(action_t.squeeze(0), i, trainer.env, assignments)
                    )

        env_actions = {
            "coordinator": coord_actions[0] if coord_actions else {},
            "coordinators": coord_actions,
            "vessels": vessel_actions,
            "ports": port_actions,
        }
        action_rows.extend(
            _build_action_trace_rows(
                t=step_i,
                policy=policy_label,
                coordinator_actions=coord_actions,
                vessel_actions=vessel_actions,
                port_actions=port_actions,
            )
        )
        obs, rewards, done, info = trainer.env.step(env_actions)
        event_rows.extend(
            _build_event_log_rows(
                default_t=step_i,
                policy=policy_label,
                info=info,
            )
        )
        step_requests = float(info.get("requests_submitted", 0.0))
        step_accepted = float(info.get("requests_accepted", 0.0))
        cumulative_vessel_requests += step_requests
        cumulative_port_accepted += step_accepted
        rows.append(
            _build_step_trace_row(
                t=step_i,
                policy=policy_label,
                rewards=rewards,
                info=info,
                vessels=trainer.env.vessels,
                ports=trainer.env.ports,
                config=trainer.cfg,
                num_coordinators=trainer.env.num_coordinators,
                cumulative_vessel_requests=cumulative_vessel_requests,
                cumulative_port_accepted=cumulative_port_accepted,
            )
        )
        if done:
            break

    trace_df = pd.DataFrame(rows)
    if return_logs:
        return trace_df, pd.DataFrame(action_rows), pd.DataFrame(event_rows)
    return trace_df


def run_policy_sweep(
    policies: list[str] | None = None,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run baseline policy comparisons with consistent seed/steps."""
    policies = policies or ["independent", "reactive", "forecast", "noiseless"]
    return {
        p: run_experiment(policy_type=p, steps=steps, seed=seed, config=config)
        for p in policies
    }


def run_horizon_sweep(
    horizons: list[int] | None = None,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> dict[int, pd.DataFrame]:
    """Run forecast horizon ablation for forecast-informed policy."""
    horizons = horizons or [6, 12, 24]
    return {
        h: run_experiment(
            policy_type="forecast",
            forecast_horizon=h,
            steps=steps,
            seed=seed,
            config=config,
        )
        for h in horizons
    }


def run_noise_sweep(
    noise_levels: list[float] | None = None,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> dict[float, pd.DataFrame]:
    """Run forecast noise ablation for forecast-informed policy."""
    noise_levels = noise_levels or [0.0, 0.3, 0.5, 1.0, 2.0]
    return {
        n: run_experiment(
            policy_type="forecast",
            forecast_noise=n,
            steps=steps,
            seed=seed,
            config=config,
        )
        for n in noise_levels
    }


def run_sharing_sweep(
    sharing_modes: dict[str, bool] | None = None,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run forecast sharing ablation."""
    sharing_modes = sharing_modes or {"shared": True, "coordinator_only": False}
    return {
        label: run_experiment(
            policy_type="forecast",
            share_forecasts=share,
            steps=steps,
            seed=seed,
            config=config,
        )
        for label, share in sharing_modes.items()
    }


def summarize_policy_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute aggregate summary table from policy sweep outputs."""
    all_results = pd.concat(results.values(), ignore_index=True)
    summary = (
        all_results.groupby("policy")
        .agg(
            {
                "avg_queue": "mean",
                "dock_utilization": "mean",
                "total_emissions_co2": "last",
                "total_fuel_used": "last",
                "on_time_rate": "mean",
                "total_ops_cost_usd": "last",
                "cost_reliability": "last",
                "avg_vessel_reward": "mean",
                "coordinator_reward": "mean",
                "policy_agreement_rate": "last",
            }
        )
        .round(3)
    )
    return summary


def save_result_dict(results: dict[Any, pd.DataFrame], out_dir: str, prefix: str) -> None:
    """Write each DataFrame from a dict into CSV files."""
    for key, df in results.items():
        safe_key = str(key).replace(" ", "_")
        df.to_csv(f"{out_dir}/{prefix}_{safe_key}.csv", index=False)


# ---------------------------------------------------------------------------
# MAPPO vs heuristic comparison
# ---------------------------------------------------------------------------


def run_mappo_comparison(
    train_iterations: int = 50,
    rollout_length: int = 64,
    eval_steps: int | None = None,
    baselines: list[str] | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    mappo_kwargs: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Train a MAPPO agent and compare its evaluation against heuristic baselines.

    Returns a dict mapping policy name → per-step metrics DataFrame,
    including a ``"mappo"`` entry from evaluation of the trained agent.

    Parameters
    ----------
    train_iterations:
        Number of MAPPO train iterations (collect + update).
    rollout_length:
        Rollout length per training iteration.
    eval_steps:
        Steps for evaluation episodes.  Defaults to ``rollout_length``.
    baselines:
        List of heuristic policy names to compare against.
    seed:
        Random seed for reproducibility.
    config:
        Environment configuration overrides.
    mappo_kwargs:
        Extra kwargs forwarded to ``MAPPOConfig``.
    """
    from .mappo import MAPPOConfig, MAPPOTrainer

    cfg = get_default_config(**(config or {}))
    eval_steps = eval_steps or rollout_length
    baselines = baselines or ["independent", "reactive", "forecast", "noiseless"]

    # --- Train MAPPO ---
    extra = dict(mappo_kwargs or {})
    extra.setdefault("rollout_length", rollout_length)
    mappo_cfg = MAPPOConfig(**extra)
    trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=seed)

    train_log: list[dict[str, float]] = []
    for iteration in range(1, train_iterations + 1):
        t_roll = time.perf_counter()
        rollout_info = trainer.collect_rollout()
        rollout_time = time.perf_counter() - t_roll
        t_upd = time.perf_counter()
        update_info = trainer.update()
        update_time = time.perf_counter() - t_upd
        row: dict[str, float] = {
            "iteration": float(iteration),
            "mean_reward": rollout_info["mean_reward"],
            "joint_mean_reward": rollout_info.get("joint_mean_reward", 0.0),
            "total_reward": rollout_info["total_reward"],
            "vessel_mean_reward": rollout_info.get("vessel_mean_reward", 0.0),
            "port_mean_reward": rollout_info.get("port_mean_reward", 0.0),
            "coordinator_mean_reward": rollout_info.get("coordinator_mean_reward", 0.0),
            "rollout_time": rollout_time,
            "update_time": update_time,
        }
        for agent_type, result in update_info.items():
            row[f"{agent_type}_value_loss"] = result.value_loss
            row[f"{agent_type}_entropy"] = result.entropy
            row[f"{agent_type}_approx_kl"] = result.approx_kl
        if update_info:
            first_result = next(iter(update_info.values()))
            row["entropy_coeff"] = first_result.entropy_coeff
        train_log.append(row)

    # --- Evaluate MAPPO ---
    results: dict[str, pd.DataFrame] = {
        "mappo": run_trained_mappo_trace(
            trainer,
            num_steps=eval_steps,
            deterministic=True,
        )
    }

    # --- Run heuristic baselines ---
    for policy in baselines:
        results[policy] = run_experiment(
            policy_type=policy, steps=eval_steps, seed=seed, config=config
        )

    # Attach training log
    results["_train_log"] = pd.DataFrame(train_log)
    return results


def run_mappo_coordinator_ablation(
    train_iterations: int = 50,
    rollout_length: int = 64,
    eval_steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    mappo_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare fully learned MAPPO against a heuristic-coordinator hybrid.

    The hybrid keeps learned vessel and port policies but replaces the
    coordinator with the forecast heuristic during both rollout collection
    and evaluation.
    """
    from .mappo import MAPPOConfig, MAPPOTrainer

    cfg = get_default_config(**(config or {}))
    eval_steps = eval_steps or rollout_length
    extra = dict(mappo_kwargs or {})
    extra.setdefault("rollout_length", rollout_length)

    rows: list[dict[str, Any]] = []
    traces: dict[str, pd.DataFrame] = {}
    variants = {
        "all_learned": False,
        "heuristic_coordinator": True,
    }

    for label, heuristic_coordinator in variants.items():
        trainer = MAPPOTrainer(
            env_config=cfg,
            mappo_config=MAPPOConfig(**extra),
            seed=seed,
        )
        for _ in range(train_iterations):
            trainer.collect_rollout(heuristic_coordinator=heuristic_coordinator)
            trainer.update()

        eval_result = trainer.evaluate(
            num_steps=eval_steps,
            deterministic=True,
            heuristic_coordinator=heuristic_coordinator,
        )
        traces[label] = run_trained_mappo_trace(
            trainer,
            num_steps=eval_steps,
            deterministic=True,
            policy_label=label,
            heuristic_coordinator=heuristic_coordinator,
        )

        history = trainer.reward_history
        row: dict[str, Any] = {
            "variant": label,
            "heuristic_coordinator": heuristic_coordinator,
            "final_mean_reward": history[-1] if history else 0.0,
            "best_mean_reward": max(history) if history else 0.0,
        }
        row.update(eval_result)
        rows.append(row)

    return pd.DataFrame(rows), traces


# ---------------------------------------------------------------------------
# MAPPO hyper-parameter sweep
# ---------------------------------------------------------------------------


def run_mappo_hyperparam_sweep(
    param_grid: dict[str, list[Any]],
    train_iterations: int = 30,
    rollout_length: int = 64,
    eval_steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    base_mappo_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Sweep over MAPPO hyper-parameter combinations.

    ``param_grid`` maps ``MAPPOConfig`` field names to lists of values.
    All combinations are evaluated (grid search) and the results are
    returned in a single DataFrame with one row per combination carrying
    the evaluation metrics and training statistics.

    Example::

        df = run_mappo_hyperparam_sweep(
            param_grid={"lr": [1e-4, 3e-4, 1e-3], "entropy_coeff": [0.01, 0.05]},
            train_iterations=30,
        )

    Returns
    -------
    pd.DataFrame
        One row per configuration.  Columns include each swept param,
        training curve summary (``final_mean_reward``, ``best_mean_reward``),
        and evaluation metrics (``total_reward``, ``mean_vessel_reward``,
        ``mean_port_reward``, ``mean_coordinator_reward``).
    """
    from .mappo import MAPPOConfig, MAPPOTrainer

    cfg = get_default_config(**(config or {}))
    eval_steps = eval_steps or rollout_length
    base = dict(base_mappo_kwargs or {})
    base.setdefault("rollout_length", rollout_length)

    # Build grid (cartesian product)
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    def _product(lists: list[list[Any]]) -> list[list[Any]]:
        if not lists:
            return [[]]
        result: list[list[Any]] = []
        for item in lists[0]:
            for rest in _product(lists[1:]):
                result.append([item, *rest])
        return result

    combos = _product(value_lists)
    rows: list[dict[str, Any]] = []

    for combo in combos:
        overrides = dict(zip(keys, combo))
        run_kwargs = {**base, **overrides}
        mappo_cfg = MAPPOConfig(**run_kwargs)
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=seed)

        # Train
        for _ in range(train_iterations):
            trainer.collect_rollout()
            trainer.update()

        # Evaluate
        eval_result = trainer.evaluate(num_steps=eval_steps, deterministic=True)
        history = trainer.reward_history

        row: dict[str, Any] = dict(overrides)
        row["final_mean_reward"] = history[-1] if history else 0.0
        row["best_mean_reward"] = max(history) if history else 0.0
        row.update(eval_result)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAPPO ablation runner
# ---------------------------------------------------------------------------


def run_mappo_ablation(
    ablations: dict[str, dict[str, Any]],
    train_iterations: int = 30,
    rollout_length: int = 64,
    eval_steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    base_mappo_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run named ablation experiments varying MAPPO configuration.

    ``ablations`` maps human-readable labels to dicts of ``MAPPOConfig``
    overrides.  Each ablation is trained and evaluated independently.

    Example::

        df = run_mappo_ablation({
            "baseline":         {},
            "no_reward_norm":   {"normalize_rewards": False},
            "high_entropy":     {"entropy_coeff": 0.05},
            "no_value_clip":    {"value_clip_eps": 0.0},
        })

    Returns
    -------
    pd.DataFrame
        One row per ablation.  Columns include ``ablation`` label,
        training summary, and evaluation metrics.
    """
    from .mappo import MAPPOConfig, MAPPOTrainer

    cfg = get_default_config(**(config or {}))
    eval_steps = eval_steps or rollout_length
    base = dict(base_mappo_kwargs or {})
    base.setdefault("rollout_length", rollout_length)

    rows: list[dict[str, Any]] = []
    for label, overrides in ablations.items():
        # Separate env-config overrides (prefixed with "env_") from
        # MAPPOConfig overrides so ablations can tweak the environment.
        env_overrides = {
            k[4:]: v for k, v in overrides.items() if k.startswith("env_")
        }
        mappo_overrides = {
            k: v for k, v in overrides.items() if not k.startswith("env_")
        }
        ablation_cfg = {**cfg, **env_overrides} if env_overrides else cfg

        run_kwargs = {**base, **mappo_overrides}
        mappo_cfg = MAPPOConfig(**run_kwargs)
        trainer = MAPPOTrainer(env_config=ablation_cfg, mappo_config=mappo_cfg, seed=seed)

        # Train
        train_log: list[dict[str, float]] = []
        for iteration in range(1, train_iterations + 1):
            rollout_info = trainer.collect_rollout()
            update_info = trainer.update()
            entry: dict[str, float] = {
                "iteration": float(iteration),
                "mean_reward": rollout_info["mean_reward"],
            }
            for agent_type, result in update_info.items():
                entry[f"{agent_type}_policy_loss"] = result.policy_loss
                entry[f"{agent_type}_value_loss"] = result.value_loss
                entry[f"{agent_type}_entropy"] = result.entropy
                entry[f"{agent_type}_grad_norm"] = result.grad_norm
            train_log.append(entry)

        # Evaluate
        eval_result = trainer.evaluate(num_steps=eval_steps, deterministic=True)
        history = trainer.reward_history
        diagnostics = trainer.get_diagnostics()

        row: dict[str, Any] = {"ablation": label}
        row["final_mean_reward"] = history[-1] if history else 0.0
        row["best_mean_reward"] = max(history) if history else 0.0
        row.update(eval_result)
        # Include final diagnostics
        for dk, dv in diagnostics.items():
            row[f"diag_{dk}"] = dv
        rows.append(row)

    return pd.DataFrame(rows)
