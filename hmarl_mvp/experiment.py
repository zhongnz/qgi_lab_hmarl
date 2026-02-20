"""Experiment runner and baseline/ablation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .config import DISTANCE_NM, SEED, get_default_config
from .dynamics import dispatch_vessel, step_ports, step_vessels
from .forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
from .metrics import (
    compute_coordination_metrics,
    compute_economic_metrics,
    compute_port_metrics,
    compute_vessel_metrics,
)
from .multi_coordinator import assign_vessels_to_coordinators
from .policies import (
    FleetCoordinatorPolicy,
    PortPolicy,
    VesselPolicy,
)
from .rewards import (
    compute_coordinator_reward,
    compute_port_reward,
    compute_vessel_reward,
)
from .scheduling import DecisionCadence
from .state import initialize_ports, initialize_vessels, make_rng

VALID_POLICIES = {"independent", "reactive", "forecast", "oracle"}


def run_experiment(
    policy_type: str = "forecast",
    forecast_horizon: int = 12,
    forecast_noise: float = 0.5,
    share_forecasts: bool = True,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
    distance_nm: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run one episode and return per-step metrics.

    policy_type:
    - independent: no coordination, no forecast usage
    - reactive: coordination using current state only (no forecast)
    - forecast: forecast-informed coordination with noisy forecasts
    - oracle: forecast-informed coordination with perfect queue knowledge
    """
    if policy_type not in VALID_POLICIES:
        raise ValueError(f"Unknown policy_type={policy_type!r}. Expected one of {VALID_POLICIES}")

    cfg = get_default_config(**(config or {}))
    steps = cfg["rollout_steps"] if steps is None else steps
    distance_nm = DISTANCE_NM if distance_nm is None else distance_nm
    rng = make_rng(seed)

    ports = initialize_ports(
        num_ports=cfg["num_ports"],
        docks_per_port=cfg["docks_per_port"],
        rng=rng,
    )
    vessels = initialize_vessels(
        num_vessels=cfg["num_vessels"],
        num_ports=cfg["num_ports"],
        nominal_speed=cfg["nominal_speed"],
        rng=rng,
    )
    num_coordinators = max(1, int(cfg.get("num_coordinators", 1)))
    coordinator_agents = [
        FleetCoordinatorAgent(config=cfg, coordinator_id=i)
        for i in range(num_coordinators)
    ]
    vessel_agents = [VesselAgent(v, cfg) for v in vessels]
    port_agents = [PortAgent(p, cfg) for p in ports]
    coordinator_policy = FleetCoordinatorPolicy(config=cfg, mode=policy_type)
    vessel_policy = VesselPolicy(config=cfg, mode=policy_type)
    port_policy = PortPolicy(config=cfg, mode=policy_type)
    medium_forecaster = MediumTermForecaster(cfg["medium_horizon_days"])
    short_forecaster = ShortTermForecaster(forecast_horizon)
    oracle_forecaster = OracleForecaster(
        medium_horizon_days=cfg["medium_horizon_days"],
        short_horizon_hours=forecast_horizon,
    )
    cadence = DecisionCadence.from_config(cfg)
    latency = cadence.message_latency_steps

    directive_queue: list[tuple[int, int, dict[str, Any]]] = []
    arrival_request_queue: list[tuple[int, int, int]] = []
    slot_response_queue: list[tuple[int, int, bool, int]] = []
    pending_port_requests: dict[int, list[int]] = {
        port_id: [] for port_id in range(cfg["num_ports"])
    }
    awaiting_slot_response: set[int] = set()
    latest_directive_by_vessel: dict[int, dict[str, Any]] = {}
    last_port_actions: list[dict[str, Any]] = [dict(a.last_action) for a in port_agents]

    log: list[dict[str, Any]] = []

    for t in range(steps):
        due = cadence.due(t)

        delivered_responses: dict[int, dict[str, Any]] = {}
        remaining_directive_queue: list[tuple[int, int, dict[str, Any]]] = []
        for deliver_step, vessel_id, directive in directive_queue:
            if deliver_step <= t:
                latest_directive_by_vessel[vessel_id] = directive
            else:
                remaining_directive_queue.append((deliver_step, vessel_id, directive))
        directive_queue = remaining_directive_queue

        remaining_arrival_queue: list[tuple[int, int, int]] = []
        for deliver_step, vessel_id, destination in arrival_request_queue:
            if deliver_step <= t:
                if 0 <= destination < cfg["num_ports"]:
                    pending_port_requests[destination].append(vessel_id)
            else:
                remaining_arrival_queue.append((deliver_step, vessel_id, destination))
        arrival_request_queue = remaining_arrival_queue

        remaining_response_queue: list[tuple[int, int, bool, int]] = []
        for deliver_step, vessel_id, accepted, destination in slot_response_queue:
            if deliver_step <= t:
                delivered_responses[vessel_id] = {
                    "accepted": bool(accepted),
                    "dest_port": int(destination),
                }
            else:
                remaining_response_queue.append((deliver_step, vessel_id, accepted, destination))
        slot_response_queue = remaining_response_queue

        if policy_type == "oracle":
            medium, short = oracle_forecaster.predict(ports)
        else:
            medium = medium_forecaster.predict(num_ports=cfg["num_ports"], rng=rng)
            short = short_forecaster.predict(num_ports=cfg["num_ports"], rng=rng)
            if policy_type == "forecast" and forecast_noise > 0:
                medium += rng.normal(0, forecast_noise, medium.shape)
                short += rng.normal(0, forecast_noise, short.shape)
                medium = np.clip(medium, 0, None)
                short = np.clip(short, 0, None)

        assignments = assign_vessels_to_coordinators(vessels, num_coordinators)
        vessel_to_coordinator = {
            vessel_id: coordinator_id
            for coordinator_id, vessel_ids in assignments.items()
            for vessel_id in vessel_ids
        }

        if due["coordinator"]:
            for coordinator_id, coordinator_agent in enumerate(coordinator_agents):
                local_ids = assignments.get(coordinator_id, [])
                vessels_by_id = {v.vessel_id: v for v in vessels}
                local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
                if not local_vessels:
                    local_vessels = vessels
                directive = coordinator_policy.act(
                    agent=coordinator_agent,
                    medium_forecast=medium,
                    vessels=local_vessels,
                    ports=ports,
                    rng=rng,
                )
                enriched = {
                    **directive,
                    "coordinator_id": coordinator_id,
                    "assigned_vessel_ids": local_ids,
                }
                for vessel_id in local_ids:
                    directive_queue.append((t + latency, vessel_id, dict(enriched)))

        forecast_for_vessels = short if share_forecasts else np.zeros_like(short)
        vessel_actions: list[dict[str, Any]] = []
        vessel_actions_for_metrics: list[dict[str, Any]] = []
        for vessel_agent in vessel_agents:
            vessel = vessel_agent.state
            vessel_id = vessel.vessel_id
            coordinator_id = vessel_to_coordinator.get(vessel_id, 0)
            fallback_directive = {
                **coordinator_agents[coordinator_id].last_action,
                "coordinator_id": coordinator_id,
                "assigned_vessel_ids": assignments.get(coordinator_id, []),
            }
            directive = latest_directive_by_vessel.get(vessel_id, fallback_directive)

            if due["vessel"]:
                action = vessel_policy.act(vessel_agent, forecast_for_vessels, directive)
            else:
                action = dict(vessel_agent.last_action)

            requested_now = False
            if (
                due["vessel"]
                and action.get("request_arrival_slot", False)
                and not vessel.at_sea
                and vessel_id not in awaiting_slot_response
            ):
                destination = int(directive.get("dest_port", vessel.destination))
                arrival_request_queue.append((t + latency, vessel_id, destination))
                awaiting_slot_response.add(vessel_id)
                requested_now = True

            response = delivered_responses.get(vessel_id)
            if response is not None:
                if response["accepted"] and not vessel.at_sea:
                    dispatch_vessel(vessel, int(response["dest_port"]), action["target_speed"], cfg)
                elif not response["accepted"] and not vessel.at_sea:
                    vessel.delay_hours += 1.0
                awaiting_slot_response.discard(vessel_id)
            elif vessel_id in awaiting_slot_response and not vessel.at_sea:
                vessel.delay_hours += 1.0

            vessel_actions.append(action)
            vessel_actions_for_metrics.append({"request_arrival_slot": requested_now})

        pre_step_at_sea = {v.vessel_id: v.at_sea for v in vessels}

        step_vessels(vessels, distance_nm=distance_nm, config=cfg, dt_hours=1.0)

        for vessel in vessels:
            if (
                pre_step_at_sea.get(vessel.vessel_id, False)
                and not vessel.at_sea
                and vessel.location == vessel.destination
            ):
                ports[vessel.location].queue += 1

        forecast_for_ports = short if share_forecasts else np.zeros_like(short)
        accepted_this_step = [0 for _ in ports]
        if due["port"]:
            incoming = sum(len(queue) for queue in pending_port_requests.values())
            current_port_actions: list[dict[str, Any]] = []
            for i, port_agent in enumerate(port_agents):
                action = port_policy.act(port_agent, incoming, forecast_for_ports[i])
                current_port_actions.append(action)
                backlog = pending_port_requests.get(i, [])
                available_slots = max(port_agent.state.docks - port_agent.state.occupied, 0)
                accept_limit = min(
                    len(backlog),
                    max(int(action.get("accept_requests", 0)), 0),
                    available_slots,
                )
                accepted = backlog[:accept_limit]
                rejected = backlog[accept_limit:]
                accepted_this_step[i] = len(accepted)
                pending_port_requests[i] = []
                response_step = t + latency
                for vessel_id in accepted:
                    slot_response_queue.append((response_step, vessel_id, True, i))
                for vessel_id in rejected:
                    slot_response_queue.append((response_step, vessel_id, False, i))
            last_port_actions = current_port_actions

        service_rates = [
            int(action.get("service_rate", 1))
            for action in (
                last_port_actions if last_port_actions else [agent.last_action for agent in port_agents]
            )
        ]
        step_ports(ports, service_rates, dt_hours=1.0)

        port_actions_for_metrics = [
            {
                **last_port_actions[i],
                "accept_requests": int(accepted_this_step[i]) if due["port"] else 0,
            }
            for i in range(len(last_port_actions))
        ]

        vessel_metrics = compute_vessel_metrics(vessels)
        port_metrics = compute_port_metrics(ports)
        coordination_metrics = compute_coordination_metrics(
            vessel_actions_for_metrics,
            port_actions_for_metrics,
        )
        economic_metrics = compute_economic_metrics(vessels, cfg)

        rewards_vessel = [compute_vessel_reward(v, cfg) for v in vessels]
        rewards_port = [compute_port_reward(p, cfg) for p in ports]
        reward_coord = compute_coordinator_reward(vessels, ports, cfg)

        log.append(
            {
                "t": t,
                "policy": policy_type,
                "forecast_horizon": forecast_horizon,
                "forecast_noise": forecast_noise,
                "share_forecasts": int(bool(share_forecasts)),
                "num_coordinators": num_coordinators,
                "coordinator_updates": int(due["coordinator"]),
                "pending_arrival_requests": float(
                    sum(len(queue) for queue in pending_port_requests.values())
                ),
                **vessel_metrics,
                **port_metrics,
                **coordination_metrics,
                **economic_metrics,
                "avg_vessel_reward": float(np.mean(rewards_vessel)) if rewards_vessel else 0.0,
                "avg_port_reward": float(np.mean(rewards_port)) if rewards_port else 0.0,
                "coordinator_reward": reward_coord,
            }
        )

    return pd.DataFrame(log)


def run_policy_sweep(
    policies: list[str] | None = None,
    steps: int | None = None,
    seed: int = SEED,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run baseline policy comparisons with consistent seed/steps."""
    policies = policies or ["independent", "reactive", "forecast", "oracle"]
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
                "policy_agreement_rate": "mean",
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


def default_config() -> dict[str, Any]:
    """Expose a mutable copy of default config."""
    return get_default_config()
