"""Tests for YAML experiment configuration, save/load, and run_from_config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmarl_mvp.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
    run_from_config,
    save_experiment_config,
)

# ------------------------------------------------------------------
# ExperimentConfig dataclass
# ------------------------------------------------------------------


class TestExperimentConfig:
    """Unit tests for ExperimentConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.name == "experiment"
        assert cfg.num_iterations == 100
        assert cfg.num_seeds == 1
        assert cfg.tensorboard is False
        assert cfg.output_dir == "runs/experiment"
        assert cfg.early_stopping_patience == 0

    def test_custom_fields(self) -> None:
        cfg = ExperimentConfig(
            name="test_run",
            description="A test experiment",
            tags=["test", "ci"],
            env={"num_vessels": 4},
            mappo={"lr": 0.001},
            num_iterations=50,
            num_seeds=3,
            seeds=[1, 2, 3],
        )
        assert cfg.name == "test_run"
        assert cfg.env["num_vessels"] == 4
        assert cfg.mappo["lr"] == 0.001
        assert cfg.num_seeds == 3
        assert cfg.seeds == [1, 2, 3]

    def test_to_dict(self) -> None:
        cfg = ExperimentConfig(name="d", env={"k": 1})
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "d"
        assert d["env"] == {"k": 1}

    def test_from_dict(self) -> None:
        data = {
            "name": "roundtrip",
            "num_iterations": 25,
            "env": {"num_ports": 3},
            "unknown_key": "ignored",
        }
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.name == "roundtrip"
        assert cfg.num_iterations == 25
        assert cfg.env == {"num_ports": 3}

    def test_roundtrip_dict(self) -> None:
        cfg = ExperimentConfig(
            name="rt",
            tags=["a"],
            mappo={"clip_eps": 0.2},
            curriculum_stages=[{"fraction": 0.5}],
        )
        cfg2 = ExperimentConfig.from_dict(cfg.to_dict())
        assert cfg2.name == cfg.name
        assert cfg2.tags == cfg.tags
        assert cfg2.mappo == cfg.mappo
        assert cfg2.curriculum_stages == cfg.curriculum_stages


# ------------------------------------------------------------------
# YAML / JSON save & load
# ------------------------------------------------------------------


class TestSaveLoad:
    """Tests for YAML and JSON serialisation."""

    def test_save_load_json(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="json_test", num_iterations=10)
        path = tmp_path / "cfg.json"
        # Force JSON by using .json extension
        save_experiment_config(cfg, path)
        loaded = load_experiment_config(path)
        assert loaded.name == "json_test"
        assert loaded.num_iterations == 10

    def test_save_load_yaml(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        cfg = ExperimentConfig(
            name="yaml_test",
            env={"num_vessels": 2},
            num_seeds=5,
        )
        path = tmp_path / "cfg.yaml"
        save_experiment_config(cfg, path)
        # Verify YAML is valid
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "yaml_test"
        # Round-trip load
        loaded = load_experiment_config(path)
        assert loaded.name == "yaml_test"
        assert loaded.env == {"num_vessels": 2}
        assert loaded.num_seeds == 5

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="nested")
        path = tmp_path / "a" / "b" / "cfg.json"
        save_experiment_config(cfg, path)
        loaded = load_experiment_config(path)
        assert loaded.name == "nested"

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment_config(tmp_path / "nope.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="Expected a dict"):
            load_experiment_config(path)


# ------------------------------------------------------------------
# run_from_config (integration)
# ------------------------------------------------------------------


class TestRunFromConfig:
    """Integration tests for run_from_config."""

    def test_single_seed_run(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="single",
            num_iterations=3,
            output_dir=str(tmp_path / "out"),
        )
        result = run_from_config(cfg)
        assert "histories" in result
        assert len(result["histories"]) == 1
        assert len(result["histories"][0]) == 3
        # Check output files were created
        assert (tmp_path / "out" / "experiment_config.yaml").exists() or (
            tmp_path / "out" / "experiment_config.json"
        ).exists()
        assert (tmp_path / "out" / "experiment_summary.json").exists()

    def test_multi_seed_run(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="multi",
            num_iterations=2,
            num_seeds=2,
            seeds=[10, 20],
            output_dir=str(tmp_path / "out"),
        )
        result = run_from_config(cfg)
        assert len(result["seeds"]) == 2
        assert len(result["histories"]) == 2
        assert "aggregate_summary" in result

    def test_with_early_stopping(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="es",
            num_iterations=20,
            early_stopping_patience=3,
            output_dir=str(tmp_path / "out"),
        )
        result = run_from_config(cfg)
        # Should complete (may or may not early stop, but shouldn't crash)
        assert len(result["histories"]) >= 1

    def test_with_mappo_config(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="custom_mappo",
            num_iterations=2,
            mappo={"lr": 0.0005, "rollout_length": 10},
            output_dir=str(tmp_path / "out"),
        )
        result = run_from_config(cfg)
        assert len(result["histories"][0]) == 2

    def test_summary_json_content(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="summary_check",
            num_iterations=2,
            output_dir=str(tmp_path / "out"),
        )
        run_from_config(cfg)
        summary_path = tmp_path / "out" / "experiment_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["name"] == "summary_check"
        assert "config" in data
