"""YAML-based experiment configuration for reproducible runs.

Supports loading / saving full experiment definitions (environment config,
MAPPO hyper-parameters, curriculum, and run settings) as YAML files.

Usage::

    # Save a config:
    cfg = ExperimentConfig(name="baseline_3seed", num_iterations=200)
    save_experiment_config(cfg, "configs/baseline.yaml")

    # Load and run:
    cfg = load_experiment_config("configs/baseline.yaml")
    result = run_from_config(cfg)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Complete experiment specification for reproducible runs.

    Groups all tunables — environment, MAPPO, curriculum, and run
    settings — into a single serialisable object.
    """

    # Metadata
    name: str = "experiment"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Environment
    env: dict[str, Any] = field(default_factory=dict)

    # MAPPO hyper-parameters
    mappo: dict[str, Any] = field(default_factory=dict)

    # Curriculum stages (list of stage dicts)
    curriculum_stages: list[dict[str, Any]] = field(default_factory=list)

    # Run settings
    num_iterations: int = 100
    num_seeds: int = 1
    seeds: list[int] | None = None
    eval_interval: int = 10
    early_stopping_patience: int = 0
    checkpoint_dir: str | None = None
    output_dir: str = "runs/experiment"

    # Logging
    tensorboard: bool = False
    console_log: bool = True
    log_every: int = 10

    # Training mode
    heuristic_coordinator: bool = False

    # Sweep: run the same experiment for each value of a parameter.
    # Example: {"parameter": "env.docks_per_port", "values": [1, 2, 3]}
    sweep: dict[str, Any] | None = None

    # Ablation: run named variants with different overrides.
    # Example: {"variants": {"freeze_coord": {"heuristic_coordinator": true}}}
    ablation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for YAML/JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        strict: bool = True,
    ) -> ExperimentConfig:
        """Construct from a plain dict (e.g. loaded from YAML).

        Parameters
        ----------
        strict:
            When True (default), raise ``KeyError`` if unknown top-level
            keys are present.  This prevents silent config typos.
            When False, unknown keys are ignored.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(k for k in data if k not in known)
        if strict and unknown:
            raise KeyError(f"Unknown experiment config keys: {unknown}")
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save an experiment config to a YAML file.

    Falls back to JSON if ``pyyaml`` is not installed, and always
    uses JSON when the path ends in ``.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()

    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return

    try:
        import yaml

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def load_experiment_config(
    path: str | Path,
    *,
    strict: bool = True,
) -> ExperimentConfig:
    """Load an experiment config from YAML or JSON.

    Supports ``.yaml``, ``.yml``, and ``.json`` extensions.

    Parameters
    ----------
    strict:
        Passed through to :meth:`ExperimentConfig.from_dict`.
        When True, unknown top-level keys raise ``KeyError``.
    """
    path = Path(path)
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            raise ImportError(
                "pyyaml is required to load YAML configs. "
                "Install with: pip install pyyaml"
            )
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        # Try YAML first, then JSON
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict from {path}, got {type(data).__name__}")

    return ExperimentConfig.from_dict(data, strict=strict)


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dotted key (e.g. 'env.docks_per_port')."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_from_config(config: ExperimentConfig) -> dict[str, Any]:
    """Execute a full experiment from an ``ExperimentConfig``.

    Handles single- and multi-seed training with optional curriculum,
    TensorBoard logging, checkpointing, sweep, and ablation modes.

    Returns
    -------
    dict
        Result dict with histories, summaries, and aggregate stats.
    """
    import copy

    # --- Sweep mode: run once per parameter value ---
    if config.sweep is not None:
        param = config.sweep["parameter"]
        values = config.sweep["values"]
        sweep_results: dict[str, Any] = {"sweep_parameter": param, "runs": {}}
        for val in values:
            sub = copy.deepcopy(config)
            sub.sweep = None
            sub.name = f"{config.name}_{param.split('.')[-1]}_{val}"
            sub.output_dir = f"{config.output_dir}/{param.split('.')[-1]}_{val}"
            cfg_dict = sub.to_dict()
            _set_nested(cfg_dict, param, val)
            sub = ExperimentConfig.from_dict(cfg_dict, strict=False)
            sweep_results["runs"][str(val)] = run_from_config(sub)
        return sweep_results

    # --- Ablation mode: run each named variant ---
    if config.ablation is not None:
        variants = config.ablation.get("variants", {})
        ablation_results: dict[str, Any] = {"variants": {}}
        for variant_name, overrides in variants.items():
            sub = copy.deepcopy(config)
            sub.ablation = None
            sub.name = f"{config.name}_{variant_name}"
            sub.output_dir = f"{config.output_dir}/{variant_name}"
            if overrides.get("heuristic_coordinator"):
                sub.heuristic_coordinator = True
            if "frozen_agent_types" in overrides:
                import warnings
                warnings.warn(
                    f"frozen_agent_types in ablation variant '{variant_name}' "
                    "is not yet supported — skipping this variant.",
                    stacklevel=2,
                )
                continue
            ablation_results["variants"][variant_name] = run_from_config(sub)
        return ablation_results

    from .curriculum import CurriculumScheduler, CurriculumStage
    from .mappo import MAPPOConfig, MAPPOTrainer, train_multi_seed

    # Build MAPPO config
    mappo_kwargs = dict(config.mappo)
    if "hidden_dims" in mappo_kwargs and isinstance(mappo_kwargs["hidden_dims"], list):
        mappo_kwargs["hidden_dims"] = list(mappo_kwargs["hidden_dims"])
    mappo_cfg = MAPPOConfig(**mappo_kwargs) if mappo_kwargs else MAPPOConfig()

    # Build curriculum (if any)
    curriculum = None
    if config.curriculum_stages:
        stages = [CurriculumStage(**s) for s in config.curriculum_stages]
        # CurriculumScheduler takes target_config + optional stages
        target_cfg = dict(config.env) if config.env else {}
        curriculum = CurriculumScheduler(
            target_config=target_cfg,
            stages=stages,
        )

    # Build env config
    env_cfg = dict(config.env) if config.env else None

    # Build TensorBoard writer (if requested)
    tb_writer = None
    if config.tensorboard:
        tb_writer = _try_create_tb_writer(config.output_dir)

    # Build log function from TensorBoard + console settings
    log_fn = None
    log_fns: list[Any] = []
    if tb_writer is not None:
        def _tb_log(it: int, entry: dict[str, Any]) -> None:
            _write_tb_scalars(tb_writer, it, entry)
        log_fns.append(_tb_log)
    if config.console_log:
        _log_every = max(config.log_every, 1)
        _name = config.name
        def _console_log(it: int, entry: dict[str, Any]) -> None:
            if it % _log_every == 0:
                mr = entry.get("mean_reward", 0.0)
                print(f"[{_name}] iter={it:>4d}  mean_reward={mr:+.4f}")
        log_fns.append(_console_log)
    if log_fns:
        def log_fn(it: int, entry: dict[str, Any]) -> None:
            for fn in log_fns:
                fn(it, entry)

    use_heuristic_coord = config.heuristic_coordinator

    # Multi-seed or single-seed
    if config.num_seeds > 1 or (config.seeds and len(config.seeds) > 1):
        result = train_multi_seed(
            env_config=env_cfg,
            mappo_config=mappo_cfg,
            num_iterations=config.num_iterations,
            seeds=config.seeds,
            num_seeds=config.num_seeds,
            curriculum=curriculum,
            eval_interval=config.eval_interval,
            early_stopping_patience=config.early_stopping_patience,
            checkpoint_dir=config.checkpoint_dir or str(
                Path(config.output_dir) / "checkpoints"
            ),
        )
    else:
        seed = (config.seeds[0] if config.seeds else 42)
        trainer = MAPPOTrainer(
            env_config=env_cfg,
            mappo_config=mappo_cfg,
            seed=seed,
        )
        trainer.heuristic_coordinator = use_heuristic_coord
        history = trainer.train(
            num_iterations=config.num_iterations,
            curriculum=curriculum,
            eval_interval=config.eval_interval,
            log_fn=log_fn,
            checkpoint_dir=config.checkpoint_dir or str(
                Path(config.output_dir) / "checkpoints"
            ),
            early_stopping_patience=config.early_stopping_patience,
        )
        result = {
            "seeds": [seed],
            "histories": [history],
            "summaries": [MAPPOTrainer.training_summary(history)],
        }

    if tb_writer is not None:
        tb_writer.close()

    # Save config alongside results for full reproducibility
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, out / "experiment_config.yaml")

    # Save per-seed metrics CSVs
    import pandas as pd

    seeds = result.get("seeds", [])
    histories = result.get("histories", [])
    for seed_val, history in zip(seeds, histories):
        seed_dir = out / f"seed_{seed_val}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if history:
            pd.DataFrame(history).to_csv(seed_dir / "metrics.csv", index=False)

    # Save summary CSV (one row per seed)
    summaries = result.get("summaries", [])
    if summaries:
        summary_rows = []
        for seed_val, s in zip(seeds, summaries):
            row = {"seed": seed_val, **s}
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)

    # Save summary JSON
    summary_data = {
        "name": config.name,
        "config": config.to_dict(),
        "summaries": summaries,
        "aggregate": result.get("aggregate_summary", {}),
    }
    with open(out / "experiment_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# TensorBoard helpers
# ---------------------------------------------------------------------------


def _try_create_tb_writer(log_dir: str) -> Any:
    """Attempt to create a TensorBoard SummaryWriter.

    Returns ``None`` if ``tensorboard`` is not installed.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(Path(log_dir) / "tb"))
    except ImportError:
        return None


def _write_tb_scalars(
    writer: Any,
    step: int,
    entry: dict[str, Any],
) -> None:
    """Write scalar metrics from a training log entry to TensorBoard."""
    _SCALAR_KEYS = {
        "mean_reward", "joint_mean_reward", "total_reward",
        "vessel_mean_reward", "port_mean_reward", "coordinator_mean_reward",
        "lr", "entropy_coeff",
        "rollout_time", "update_time", "iter_time",
    }
    _SUFFIX_KEYS = {
        "_policy_loss", "_value_loss", "_entropy",
        "_clip_frac", "_grad_norm", "_approx_kl", "_explained_variance",
    }

    for key, val in entry.items():
        if not isinstance(val, (int, float)):
            continue
        if key in _SCALAR_KEYS or any(key.endswith(s) for s in _SUFFIX_KEYS):
            writer.add_scalar(f"train/{key}", val, step)

    writer.flush()
