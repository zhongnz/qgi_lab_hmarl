"""Tests for typed configuration schema and validation behavior."""

from __future__ import annotations

import unittest

from hmarl_mvp.config import (
    HMARLConfig,
    get_default_config,
    resolve_distance_matrix,
    validate_config,
)


class ConfigSchemaTests(unittest.TestCase):
    def test_default_schema_round_trip(self) -> None:
        cfg = HMARLConfig().to_dict()
        self.assertEqual(cfg["num_ports"], 3)
        self.assertEqual(cfg["rollout_steps"], 138)

    def test_get_default_config_with_valid_overrides(self) -> None:
        cfg = get_default_config(
            speed_min=9.0,
            nominal_speed=10.0,
            speed_max=12.0,
            num_vessels=4,
            num_coordinators=2,
        )
        self.assertEqual(cfg["nominal_speed"], 10.0)
        self.assertEqual(cfg["num_coordinators"], 2)

    def test_invalid_nominal_speed_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(speed_min=10.0, nominal_speed=20.0, speed_max=18.0)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_default_config(not_a_real_field=1)

    def test_non_positive_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(message_latency_steps=0)

    def test_validate_config_allows_partial_mapping(self) -> None:
        cfg = validate_config({"rollout_steps": 5, "num_coordinators": 1})
        self.assertEqual(cfg["rollout_steps"], 5)
        self.assertEqual(cfg["num_coordinators"], 1)

    def test_single_mission_config_validates(self) -> None:
        cfg = get_default_config(episode_mode="single_mission", mission_success_on="arrival")
        self.assertEqual(cfg["episode_mode"], "single_mission")
        self.assertEqual(cfg["mission_success_on"], "arrival")
        self.assertEqual(cfg["forecast_source"], "ground_truth")

    def test_ground_truth_forecast_source_validates(self) -> None:
        cfg = get_default_config(forecast_source="ground_truth")
        self.assertEqual(cfg["forecast_source"], "ground_truth")

    def test_invalid_episode_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(episode_mode="not_real")

    def test_invalid_mission_success_on_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(mission_success_on="dock")

    def test_invalid_forecast_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_default_config(forecast_source="not_real")

    def test_default_distances_are_reachable_within_episode(self) -> None:
        cfg = get_default_config()
        distance_nm = resolve_distance_matrix(cfg["num_ports"])
        travel_hours = distance_nm[distance_nm > 0] / cfg["nominal_speed"]
        self.assertLessEqual(float(travel_hours.max()), float(cfg["rollout_steps"]))


if __name__ == "__main__":
    unittest.main()
