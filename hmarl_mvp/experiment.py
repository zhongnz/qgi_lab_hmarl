"""Experiment runner and baseline/ablation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SEED, get_default_config
from .env import MaritimeEnv
from .forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
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

VALID_POLICIES = {"independent", "reactive", "forecast", "oracle", "learned_forecast"}


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
    - oracle: forecast-informed coordination with perfect queue knowledge
    - learned_forecast: uses a trained ``LearnedForecaster`` for predictions
    """
    if policy_type not in VALID_POLICIES:
        raise ValueError(f"Unknown policy_type={policy_type!r}. Expected one of {VALID_POLICIES}")
    if policy_type == "learned_forecast" and learned_forecaster is None:
        raise ValueError("learned_forecaster must be provided when policy_type='learned_forecast'")

    # For learned_forecast, use the same heuristic policy logic but with learned predictions
    effective_mode = "forecast" if policy_type == "learned_forecast" else policy_type

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
        elif policy_type == "learned_forecast" and learned_forecaster is not None:
            # Use the trained model for both medium and short forecasts
            medium = learned_forecaster.predict(env.ports, rng=rng)
            short = learned_forecaster.predict(env.ports, rng=rng)
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
        step_economic = compute_economic_step_deltas(
            step_fuel_used=float(info.get("step_fuel_used", 0.0)),
            step_co2_emitted=float(info.get("step_co2_emitted", 0.0)),
            step_delay_hours=float(info.get("step_delay_hours", 0.0)),
            config=cfg,
        )
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
                "weather_enabled": int(bool(info.get("weather_enabled", False))),
                "mean_sea_state": float(info.get("mean_sea_state", 0.0)),
                "max_sea_state": float(info.get("max_sea_state", 0.0)),
                **vessel_metrics,
                **port_metrics,
                **coordination_metrics,
                **economic_metrics,
                **step_economic,
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
    baselines = baselines or ["independent", "reactive", "forecast", "oracle"]

    # --- Train MAPPO ---
    extra = dict(mappo_kwargs or {})
    extra.setdefault("rollout_length", rollout_length)
    mappo_cfg = MAPPOConfig(**extra)
    trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=seed)

    train_log: list[dict[str, float]] = []
    for iteration in range(1, train_iterations + 1):
        rollout_info = trainer.collect_rollout()
        update_info = trainer.update()
        row: dict[str, float] = {
            "iteration": float(iteration),
            "mean_reward": rollout_info["mean_reward"],
        }
        for agent_type, result in update_info.items():
            row[f"{agent_type}_value_loss"] = result.value_loss
        train_log.append(row)

    # --- Evaluate MAPPO ---
    from .metrics import compute_economic_step_deltas as _ced

    mappo_rows: list[dict[str, Any]] = []
    obs = trainer.env.reset()

    import torch

    vessel_ac = trainer.actor_critics["vessel"]
    port_ac = trainer.actor_critics["port"]
    coord_ac = trainer.actor_critics["coordinator"]
    vessel_ac.eval()
    port_ac.eval()
    coord_ac.eval()
    device = trainer.device

    from .mappo import _nn_to_coordinator_action, _nn_to_port_action, _nn_to_vessel_action

    for step_i in range(eval_steps):
        global_state = trainer.env.get_global_state()
        gs_tensor = torch.as_tensor(
            global_state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            vessel_actions = []
            for v_obs in obs["vessels"]:
                v_obs_n = trainer._eval_normalize_obs(v_obs, "vessel")
                v_t = torch.as_tensor(v_obs_n, dtype=torch.float32, device=device).unsqueeze(0)
                a, _, _ = vessel_ac.get_action_and_value(v_t, gs_tensor, deterministic=True)
                vessel_actions.append(_nn_to_vessel_action(a.squeeze(0), trainer.cfg))

            port_actions = []
            for i, p_obs in enumerate(obs["ports"]):
                p_obs_n = trainer._eval_normalize_obs(p_obs, "port")
                p_t = torch.as_tensor(p_obs_n, dtype=torch.float32, device=device).unsqueeze(0)
                p_mask = trainer._port_mask_tensor(i)
                a, _, _ = port_ac.get_action_and_value(
                    p_t, gs_tensor, deterministic=True, action_mask=p_mask
                )
                port_actions.append(_nn_to_port_action(a.squeeze(0), i, trainer.env))

            assignments = trainer.env._build_assignments()
            coord_actions = []
            for i, c_obs in enumerate(obs["coordinators"]):
                c_obs_n = trainer._eval_normalize_obs(c_obs, "coordinator")
                c_t = torch.as_tensor(c_obs_n, dtype=torch.float32, device=device).unsqueeze(0)
                a, _, _ = coord_ac.get_action_and_value(c_t, gs_tensor, deterministic=True)
                coord_actions.append(
                    _nn_to_coordinator_action(a.squeeze(0), i, trainer.env, assignments)
                )

        env_actions = {
            "coordinator": coord_actions[0] if coord_actions else {},
            "coordinators": coord_actions,
            "vessels": vessel_actions,
            "ports": port_actions,
        }
        obs, rewards, done, info = trainer.env.step(env_actions)
        vessel_metrics = compute_vessel_metrics(trainer.env.vessels)
        port_metrics = compute_port_metrics(trainer.env.ports)
        economic_metrics = compute_economic_metrics(trainer.env.vessels, trainer.cfg)
        step_economic = _ced(
            step_fuel_used=float(info.get("step_fuel_used", 0.0)),
            step_co2_emitted=float(info.get("step_co2_emitted", 0.0)),
            step_delay_hours=float(info.get("step_delay_hours", 0.0)),
            config=trainer.cfg,
        )
        mappo_rows.append({
            "t": step_i,
            "policy": "mappo",
            **vessel_metrics,
            **port_metrics,
            **economic_metrics,
            **step_economic,
            "avg_vessel_reward": float(np.mean(rewards["vessels"])),
            "avg_port_reward": float(np.mean(rewards["ports"])),
            "coordinator_reward": float(rewards["coordinator"]),
        })
        if done:
            break

    results: dict[str, pd.DataFrame] = {"mappo": pd.DataFrame(mappo_rows)}

    # --- Run heuristic baselines ---
    for policy in baselines:
        results[policy] = run_experiment(
            policy_type=policy, steps=eval_steps, seed=seed, config=config
        )

    # Attach training log
    results["_train_log"] = pd.DataFrame(train_log)
    return results


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
