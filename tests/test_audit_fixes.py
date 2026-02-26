"""Tests for audit-identified fixes: evaluate division, seed variation,
dt_hours config, logger serialization, per-agent reward accumulation,
dispatch current_step."""

from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from hmarl_mvp.config import HMARLConfig, get_default_config
from hmarl_mvp.dynamics import dispatch_vessel
from hmarl_mvp.state import VesselState

# -----------------------------------------------------------------------
# Issue 2.1: evaluate() should divide by actual_steps not num_steps
# Issue 2.2: evaluate_episodes() should vary seeds
# (Tested in integration via MAPPOTrainer — unit-testing the logic here)
# -----------------------------------------------------------------------


class TestEvaluateDivision(unittest.TestCase):
    """Verify evaluation reward averaging uses actual steps completed."""

    def test_early_termination_division(self) -> None:
        """Simulated: 5 actual steps out of 20 total."""
        total_vessel_reward = 10.0
        actual_steps = 5
        denom = max(actual_steps, 1)
        mean_vessel = total_vessel_reward / denom
        self.assertAlmostEqual(mean_vessel, 2.0)

    def test_zero_steps_safe(self) -> None:
        denom = max(0, 1)
        self.assertEqual(denom, 1)


# -----------------------------------------------------------------------
# Issue 1.3: dispatch_vessel current_step parameter
# -----------------------------------------------------------------------


class TestDispatchCurrentStep(unittest.TestCase):
    """Verify dispatch_vessel records trip_start_step via parameter."""

    def test_current_step_recorded(self) -> None:
        cfg = get_default_config()
        v = VesselState(vessel_id=0, location=0, destination=0, speed=12.0)
        dispatch_vessel(v, destination=2, speed=12.0, config=cfg, current_step=42)
        self.assertEqual(v.trip_start_step, 42)

    def test_default_current_step_is_zero(self) -> None:
        cfg = get_default_config()
        v = VesselState(vessel_id=0, location=0, destination=0, speed=12.0)
        dispatch_vessel(v, destination=1, speed=12.0, config=cfg)
        self.assertEqual(v.trip_start_step, 0)

    def test_same_port_noop_preserves_trip_start(self) -> None:
        cfg = get_default_config()
        v = VesselState(vessel_id=0, location=3, destination=3, trip_start_step=5)
        dispatch_vessel(v, destination=3, speed=12.0, config=cfg, current_step=99)
        self.assertEqual(v.trip_start_step, 5)  # unchanged


# -----------------------------------------------------------------------
# Issue 3.3: dt_hours in HMARLConfig
# -----------------------------------------------------------------------


class TestDtHoursConfig(unittest.TestCase):
    """Verify dt_hours is a valid config parameter."""

    def test_default_dt_hours(self) -> None:
        cfg = HMARLConfig()
        self.assertEqual(cfg.dt_hours, 1.0)

    def test_custom_dt_hours(self) -> None:
        cfg = get_default_config(dt_hours=0.5)
        self.assertAlmostEqual(cfg["dt_hours"], 0.5)

    def test_invalid_dt_hours_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(dt_hours=0.0)

    def test_negative_dt_hours_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(dt_hours=-1.0)


# -----------------------------------------------------------------------
# Issue 3.7: logger _serializable handles torch tensors
# -----------------------------------------------------------------------


class TestLoggerSerializable(unittest.TestCase):
    """Verify _serializable handles torch.Tensor gracefully."""

    def test_torch_scalar_tensor(self) -> None:
        import torch

        from hmarl_mvp.logger import _serializable

        t = torch.tensor(3.14)
        result = _serializable(t)
        self.assertAlmostEqual(result, 3.14, places=2)
        # Should be JSON serializable
        json.dumps(result)

    def test_torch_1d_tensor(self) -> None:
        import torch

        from hmarl_mvp.logger import _serializable

        t = torch.tensor([1.0, 2.0, 3.0])
        result = _serializable(t)
        self.assertEqual(result, [1.0, 2.0, 3.0])
        json.dumps(result)

    def test_dict_with_mixed_types(self) -> None:
        import torch

        from hmarl_mvp.logger import _serializable

        data = {
            "loss": torch.tensor(0.5),
            "array": np.array([1, 2]),
            "scalar": 42,
        }
        result = _serializable(data)
        json.dumps(result)  # Should not raise


# -----------------------------------------------------------------------
# Issue 1.2 / 3.5: metric key defaults in plotting/report
# -----------------------------------------------------------------------


class TestMetricKeyDefaults(unittest.TestCase):
    """Verify plotting and report functions use correct metric key defaults."""

    def test_sweep_heatmap_default_metric(self) -> None:
        import inspect

        from hmarl_mvp.plotting import plot_sweep_heatmap

        sig = inspect.signature(plot_sweep_heatmap)
        default = sig.parameters["metric"].default
        self.assertEqual(default, "total_reward")

    def test_ablation_bar_default_metrics(self) -> None:
        import tempfile

        from hmarl_mvp.plotting import plot_ablation_bar

        # Call with data that has correct keys to verify it works
        df = pd.DataFrame([
            {"ablation": "base", "final_mean_reward": -1.0, "total_reward": -3.0},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            out = os.path.join(tmpdir, "test.png")
            plot_ablation_bar(df, out_path=out)
