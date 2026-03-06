"""Tests for YAML experiment configuration, save/load, and run_from_config."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

import hmarl_mvp.experiment_config as experiment_config_mod
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
        with pytest.raises(KeyError, match="Unknown experiment config keys"):
            ExperimentConfig.from_dict(data)

    def test_from_dict_non_strict_ignores_unknown(self) -> None:
        data = {
            "name": "roundtrip",
            "num_iterations": 25,
            "env": {"num_ports": 3},
            "unknown_key": "ignored",
        }
        cfg = ExperimentConfig.from_dict(data, strict=False)
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

    def test_load_unknown_top_level_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": "x", "unknown_key": 1}))
        with pytest.raises(KeyError, match="Unknown experiment config keys"):
            load_experiment_config(path)

    def test_load_unknown_top_level_key_non_strict(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": "x", "unknown_key": 1}))
        loaded = load_experiment_config(path, strict=False)
        assert loaded.name == "x"

    def test_save_yaml_falls_back_to_json_without_pyyaml(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        orig_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("pyyaml unavailable")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        path = tmp_path / "cfg.yaml"
        save_experiment_config(ExperimentConfig(name="fallback"), path)
        fallback_path = tmp_path / "cfg.json"
        assert fallback_path.exists()

    def test_load_yaml_without_pyyaml_raises(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        orig_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("pyyaml unavailable")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        path = tmp_path / "cfg.yaml"
        path.write_text("name: demo\n")
        with pytest.raises(ImportError, match="pyyaml is required"):
            load_experiment_config(path)

    def test_load_unknown_suffix_json_fallback_without_yaml(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        orig_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("pyyaml unavailable")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        path = tmp_path / "cfg.txt"
        path.write_text(json.dumps({"name": "txt_config", "num_iterations": 3}))
        loaded = load_experiment_config(path)
        assert loaded.name == "txt_config"
        assert loaded.num_iterations == 3


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

    def test_tensorboard_path_and_curriculum_branch(self, monkeypatch: Any, tmp_path: Path) -> None:
        class DummyWriter:
            def __init__(self) -> None:
                self.closed = False

            def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
                return None

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        writer = DummyWriter()
        monkeypatch.setattr(
            experiment_config_mod,
            "_try_create_tb_writer",
            lambda _log_dir: writer,
        )
        cfg = ExperimentConfig(
            name="tb_curriculum",
            num_iterations=1,
            tensorboard=True,
            mappo={"hidden_dims": [32, 32]},
            env={"num_ports": 2, "num_vessels": 3, "rollout_steps": 12},
            curriculum_stages=[
                {"fraction": 0.0, "config_overrides": {"weather_enabled": False}},
                {"fraction": 0.5, "config_overrides": {"weather_enabled": True}},
            ],
            output_dir=str(tmp_path / "out"),
        )
        run_from_config(cfg)
        assert writer.closed is True


class TestTensorboardHelpers:
    def test_try_create_tb_writer_import_error(self, monkeypatch: Any) -> None:
        orig_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("torch.utils.tensorboard"):
                raise ImportError("tensorboard unavailable")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert experiment_config_mod._try_create_tb_writer("runs/tb_test") is None

    def test_write_tb_scalars_logs_supported_keys(self) -> None:
        class DummyWriter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, float, int]] = []
                self.flushed = False

            def add_scalar(self, name: str, value: float, step: int) -> None:
                self.calls.append((name, float(value), step))

            def flush(self) -> None:
                self.flushed = True

        writer = DummyWriter()
        experiment_config_mod._write_tb_scalars(
            writer=writer,
            step=7,
            entry={
                "mean_reward": 1.2,
                "vessel_value_loss": 0.5,
                "coordinator_approx_kl": 0.01,
                "non_numeric": "x",
            },
        )
        logged_names = {name for name, _value, _step in writer.calls}
        assert "train/mean_reward" in logged_names
        assert "train/vessel_value_loss" in logged_names
        assert "train/coordinator_approx_kl" in logged_names
        assert writer.flushed is True
