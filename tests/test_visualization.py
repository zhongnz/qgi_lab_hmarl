"""Tests for animated replay and reward decomposition visualizations."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")

from hmarl_mvp.plotting import (
    collect_episode_snapshots,
    plot_animated_replay,
    plot_reward_decomposition,
)


class CollectEpisodeSnapshotsTests(unittest.TestCase):
    """Tests for the episode snapshot collector."""

    def test_basic_collection(self) -> None:
        snaps = collect_episode_snapshots(steps=5, seed=42)
        self.assertEqual(len(snaps), 5)
        self.assertIsInstance(snaps, list)

    def test_snapshot_structure(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        s = snaps[0]
        required_keys = {
            "t", "vessels", "ports", "weather", "distance_nm",
            "rewards", "reward_components", "coordinator_action",
            "info", "done",
        }
        self.assertTrue(required_keys.issubset(s.keys()), f"Missing keys: {required_keys - s.keys()}")

    def test_vessel_snapshot_fields(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        v = snaps[0]["vessels"][0]
        expected_fields = {
            "vessel_id", "location", "destination", "position_nm",
            "leg_distance", "progress", "speed", "fuel", "initial_fuel",
            "fuel_frac", "cumulative_fuel_used", "emissions",
            "at_sea", "stalled", "pending_departure", "depart_at_step",
            "port_service_state", "delay_hours", "schedule_delay_hours",
            "last_schedule_delay_hours", "requested_arrival_time",
            "completed_arrivals", "completed_scheduled_arrivals",
            "on_time_arrivals", "mission_done", "mission_success",
            "mission_failed",
        }
        self.assertTrue(expected_fields.issubset(v.keys()))

    def test_port_snapshot_fields(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        p = snaps[0]["ports"][0]
        expected_fields = {
            "port_id", "queue", "docks", "occupied", "available",
            "utilization", "cumulative_wait_hours", "vessels_served",
            "queued_vessel_ids", "servicing_vessel_ids",
        }
        self.assertTrue(expected_fields.issubset(p.keys()))

    def test_reward_components_present(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        comp = snaps[0]["reward_components"]
        self.assertIn("vessel", comp)
        self.assertIn("port", comp)
        self.assertIn("coordinator", comp)
        self.assertIn("fuel_cost", comp["vessel"])
        self.assertIn("wait_penalty", comp["port"])
        self.assertIn("queue_penalty", comp["coordinator"])

    def test_actions_captured(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        actions = snaps[0]["actions"]
        # Coordinator
        self.assertIn("dest_port", actions["coordinator"])
        self.assertIn("departure_window_hours", actions["coordinator"])
        self.assertIn("emission_budget", actions["coordinator"])
        # Vessels
        self.assertEqual(len(actions["vessels"]), 8)
        self.assertIn("target_speed", actions["vessels"][0])
        self.assertIn("request_arrival_slot", actions["vessels"][0])
        # Ports
        self.assertEqual(len(actions["ports"]), 5)
        self.assertIn("service_rate", actions["ports"][0])

    def test_fleet_kpis_present(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        kpis = snaps[0]["fleet_kpis"]
        for key in ("on_time_rate", "queue_imbalance_std", "fleet_emissions",
                     "fleet_fuel_used", "stalled_count",
                     "vessel_utilization_rate"):
            self.assertIn(key, kpis)

    def test_events_list_present(self) -> None:
        snaps = collect_episode_snapshots(steps=5, seed=42)
        # At least one snapshot should have events (dispatches, arrivals, etc.)
        has_events = any(len(s.get("events", [])) > 0 for s in snaps)
        self.assertTrue(has_events, "Expected at least one snapshot with events")

    def test_weather_enabled_by_default(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        self.assertIsNotNone(snaps[0]["weather"])

    def test_weather_enabled(self) -> None:
        snaps = collect_episode_snapshots(
            steps=3, seed=42,
            config={"weather_enabled": True, "weather_autocorrelation": 0.7},
        )
        self.assertIsNotNone(snaps[0]["weather"])
        self.assertEqual(snaps[0]["weather"].shape, (5, 5))

    def test_deterministic_with_seed(self) -> None:
        s1 = collect_episode_snapshots(steps=5, seed=123)
        s2 = collect_episode_snapshots(steps=5, seed=123)
        for a, b in zip(s1, s2):
            self.assertEqual(a["rewards"]["coordinator"], b["rewards"]["coordinator"])

    def test_step_indices_sequential(self) -> None:
        snaps = collect_episode_snapshots(steps=5, seed=42)
        ts = [s["t"] for s in snaps]
        self.assertEqual(ts, list(range(5)))

    def test_correct_agent_counts(self) -> None:
        snaps = collect_episode_snapshots(steps=3, seed=42)
        self.assertEqual(len(snaps[0]["vessels"]), 8)
        self.assertEqual(len(snaps[0]["ports"]), 5)

    def test_custom_config(self) -> None:
        snaps = collect_episode_snapshots(
            steps=3, seed=42,
            config={"num_vessels": 4, "num_ports": 3},
        )
        self.assertEqual(len(snaps[0]["vessels"]), 4)
        self.assertEqual(len(snaps[0]["ports"]), 3)

    def test_fuel_frac_in_range(self) -> None:
        snaps = collect_episode_snapshots(steps=5, seed=42)
        for s in snaps:
            for v in s["vessels"]:
                self.assertGreaterEqual(v["fuel_frac"], 0.0)
                self.assertLessEqual(v["fuel_frac"], 1.0 + 1e-9)


class PlotAnimatedReplayTests(unittest.TestCase):
    """Tests for the animated spatial replay."""

    snaps: ClassVar[list[dict[str, Any]]]
    weather_snaps: ClassVar[list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.snaps = collect_episode_snapshots(steps=5, seed=42)
        cls.weather_snaps = collect_episode_snapshots(
            steps=5, seed=42,
            config={"weather_enabled": True},
        )

    def test_returns_animation(self) -> None:
        from matplotlib.animation import FuncAnimation

        anim = plot_animated_replay(self.snaps)
        self.assertIsInstance(anim, FuncAnimation)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_save_gif(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            plot_animated_replay(self.snaps, out_path=path, fps=2)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)
        finally:
            os.unlink(path)

    def test_weather_overlay(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            plot_animated_replay(self.weather_snaps, out_path=path, fps=2)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)
        finally:
            os.unlink(path)

    def test_empty_snapshots_raises(self) -> None:
        with self.assertRaises(ValueError):
            plot_animated_replay([])


class PlotRewardDecompositionTests(unittest.TestCase):
    """Tests for the reward decomposition stacked-area chart."""

    snaps: ClassVar[list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.snaps = collect_episode_snapshots(steps=8, seed=42)

    def test_saves_png(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_reward_decomposition(self.snaps, out_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 5000)
        finally:
            os.unlink(path)

    def test_empty_snapshots_no_crash(self) -> None:
        plot_reward_decomposition([])

    def test_with_weather(self) -> None:
        snaps = collect_episode_snapshots(
            steps=5, seed=42,
            config={"weather_enabled": True, "weather_autocorrelation": 0.7},
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_reward_decomposition(snaps, out_path=path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_no_save_shows(self) -> None:
        import matplotlib.pyplot as plt

        plot_reward_decomposition(self.snaps)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
