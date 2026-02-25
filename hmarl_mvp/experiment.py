"""Experiment runner and baseline/ablation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SEED, get_default_config
from .env import MaritimeEnv
from .forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
from .metrics import (
    compute_economic_metrics,
    compute_port_metrics,
    compute_vessel_metrics,
)
from .policies import (
    FleetCoordinatorPolicy,
    PortPolicy,
    VesselPolicy,
)
from .state import make_rng

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
    cfg["rollout_steps"] = int(steps)
    rng = make_rng(seed)
    env = MaritimeEnv(config=cfg, seed=seed, distance_nm=distance_nm)
    env.reset()

    num_coordinators = env.num_coordinators
    coordinator_policy = FleetCoordinatorPolicy(config=cfg, mode=policy_type)
    vessel_policy = VesselPolicy(config=cfg, mode=policy_type)
    port_policy = PortPolicy(config=cfg, mode=policy_type)
    medium_forecaster = MediumTermForecaster(cfg["medium_horizon_days"])
    short_forecaster = ShortTermForecaster(forecast_horizon)
    oracle_forecaster = OracleForecaster(
        medium_horizon_days=cfg["medium_horizon_days"],
        short_horizon_hours=forecast_horizon,
    )

    log: list[dict[str, Any]] = []
    cumulative_vessel_requests = 0.0
    cumulative_port_accepted = 0.0

    for _ in range(steps):
        t = env.t
        step_context = env.peek_step_context()
        if policy_type == "oracle":
            medium, short = oracle_forecaster.predict(env.ports)
        else:
            medium = medium_forecaster.predict(num_ports=cfg["num_ports"], rng=rng)
            short = short_forecaster.predict(num_ports=cfg["num_ports"], rng=rng)
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
                )
            )

        forecast_for_vessels = short if share_forecasts else np.zeros_like(short)
        vessel_actions: list[dict[str, Any]] = []
        for vessel_agent in env.vessel_agents:
            vessel = vessel_agent.state
            directive = env.get_directive_for_vessel(
                vessel_id=vessel.vessel_id,
                assignments=assignments,
                latest_directive_by_vessel=latest_directive_by_vessel,
            )
            vessel_actions.append(
                vessel_policy.propose_action(
                    short_forecast=forecast_for_vessels,
                    directive=directive,
                )
            )

        forecast_for_ports = short if share_forecasts else np.zeros_like(short)
        port_actions = [
            port_policy.propose_action(
                port_state=port_agent.state,
                incoming_requests=len(pending_port_requests.get(i, [])),
                short_forecast_row=forecast_for_ports[i],
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

        vessel_metrics = compute_vessel_metrics(env.vessels)
        port_metrics = compute_port_metrics(env.ports)
        economic_metrics = compute_economic_metrics(env.vessels, cfg)
        step_requests = float(info.get("requests_submitted", 0.0))
        step_accepted = float(info.get("requests_accepted", 0.0))
        cumulative_vessel_requests += step_requests
        cumulative_port_accepted += step_accepted
        coordination_metrics = {
            "step_vessel_requests": step_requests,
            "step_port_accepted": step_accepted,
            "total_vessel_requests": cumulative_vessel_requests,
            "total_port_accepted": cumulative_port_accepted,
            "policy_agreement_rate": (
                cumulative_port_accepted / cumulative_vessel_requests
                if cumulative_vessel_requests > 0
                else 0.0
            ),
        }

        log.append(
            {
                "t": t,
                "policy": policy_type,
                "forecast_horizon": forecast_horizon,
                "forecast_noise": forecast_noise,
                "share_forecasts": int(bool(share_forecasts)),
                "num_coordinators": num_coordinators,
                "coordinator_updates": int(info["cadence_due"]["coordinator"]),
                "pending_arrival_requests": float(info.get("pending_arrival_requests", 0.0)),
                **vessel_metrics,
                **port_metrics,
                **coordination_metrics,
                **economic_metrics,
                "avg_vessel_reward": float(np.mean(rewards["vessels"])) if rewards["vessels"] else 0.0,
                "avg_port_reward": float(np.mean(rewards["ports"])) if rewards["ports"] else 0.0,
                "coordinator_reward": float(rewards["coordinator"]),
            }
        )
        if done:
            break

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


def default_config() -> dict[str, Any]:
    """Expose a mutable copy of default config."""
    return get_default_config()
