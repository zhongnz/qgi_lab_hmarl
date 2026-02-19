"""Experiment runner and baseline/ablation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DISTANCE_NM, SEED, get_default_config
from .dynamics import dispatch_vessel, step_ports, step_vessels
from .forecasts import medium_term_forecast, oracle_forecasts, short_term_forecast
from .metrics import (
    compute_coordination_metrics,
    compute_economic_metrics,
    compute_port_metrics,
    compute_vessel_metrics,
)
from .policies import (
    fleet_coordinator_policy,
    independent_port_policy,
    independent_vessel_policy,
    port_policy,
    reactive_coordinator_policy,
    reactive_port_policy,
    reactive_vessel_policy,
    vessel_policy,
)
from .rewards import (
    compute_coordinator_reward,
    compute_port_reward,
    compute_vessel_reward,
)
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

    log: list[dict[str, Any]] = []

    for t in range(steps):
        if policy_type == "oracle":
            medium, short = oracle_forecasts(
                ports=ports,
                medium_horizon_days=cfg["medium_horizon_days"],
                short_horizon_hours=forecast_horizon,
            )
        else:
            medium = medium_term_forecast(
                num_ports=cfg["num_ports"],
                horizon_days=cfg["medium_horizon_days"],
                rng=rng,
            )
            short = short_term_forecast(
                num_ports=cfg["num_ports"],
                horizon_hours=forecast_horizon,
                rng=rng,
            )
            if policy_type == "forecast" and forecast_noise > 0:
                medium += rng.normal(0, forecast_noise, medium.shape)
                short += rng.normal(0, forecast_noise, short.shape)
                medium = np.clip(medium, 0, None)
                short = np.clip(short, 0, None)

        if policy_type == "independent":
            directive = {
                "dest_port": int(rng.integers(0, cfg["num_ports"])),
                "departure_window_hours": 12,
                "emission_budget": 50.0,
            }
        elif policy_type == "reactive":
            directive = reactive_coordinator_policy(ports)
        else:
            directive = fleet_coordinator_policy(medium, vessels)

        if policy_type == "independent":
            vessel_actions = [independent_vessel_policy(cfg) for _ in vessels]
        elif policy_type == "reactive":
            vessel_actions = [reactive_vessel_policy(cfg) for _ in vessels]
        else:
            forecast_for_vessels = short if share_forecasts else np.zeros_like(short)
            vessel_actions = [
                vessel_policy(v, forecast_for_vessels, directive, cfg) for v in vessels
            ]

        for vessel, action in zip(vessels, vessel_actions):
            if not vessel.at_sea:
                dispatch_vessel(vessel, directive["dest_port"], action["target_speed"], cfg)

        step_vessels(vessels, distance_nm=distance_nm, config=cfg, dt_hours=1.0)

        for vessel in vessels:
            if not vessel.at_sea and vessel.location == vessel.destination:
                ports[vessel.location].queue += 1

        incoming = sum(1 for a in vessel_actions if a.get("request_arrival_slot", False))
        if policy_type == "independent":
            port_actions = [independent_port_policy() for _ in ports]
        elif policy_type == "reactive":
            port_actions = [reactive_port_policy(p) for p in ports]
        else:
            forecast_for_ports = short if share_forecasts else np.zeros_like(short)
            port_actions = [
                port_policy(p, incoming, forecast_for_ports[i]) for i, p in enumerate(ports)
            ]
        step_ports(ports, [a["service_rate"] for a in port_actions], dt_hours=1.0)

        vessel_metrics = compute_vessel_metrics(vessels)
        port_metrics = compute_port_metrics(ports)
        coordination_metrics = compute_coordination_metrics(vessel_actions, port_actions)
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
