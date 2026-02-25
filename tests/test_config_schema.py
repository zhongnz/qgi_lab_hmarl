"""Tests for typed configuration schema and validation behavior."""

from __future__ import annotations

import unittest

from hmarl_mvp.config import HMARLConfig, get_default_config, validate_config


class ConfigSchemaTests(unittest.TestCase):
    def test_default_schema_round_trip(self) -> None:
        cfg = HMARLConfig().to_dict()
        self.assertEqual(cfg["num_ports"], 5)
        self.assertEqual(cfg["rollout_steps"], 20)

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


if __name__ == "__main__":
    unittest.main()
