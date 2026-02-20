"""Tests for async cadence and multi-coordinator scaffolding."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.multi_coordinator import (
    assign_vessels_to_coordinators,
    build_multi_coordinator_directives,
)
from hmarl_mvp.scheduling import DecisionCadence, should_update
from hmarl_mvp.state import initialize_vessels, make_rng


class ArchitectureScaffoldingTests(unittest.TestCase):
    def test_should_update(self) -> None:
        self.assertTrue(should_update(0, 3))
        self.assertFalse(should_update(1, 3))
        self.assertTrue(should_update(3, 3))

    def test_decision_cadence_from_config(self) -> None:
        cfg = get_default_config(
            coord_decision_interval_steps=12,
            vessel_decision_interval_steps=1,
            port_decision_interval_steps=2,
            message_latency_steps=2,
        )
        cadence = DecisionCadence.from_config(cfg)
        due_t0 = cadence.due(0)
        due_t1 = cadence.due(1)
        due_t2 = cadence.due(2)

        self.assertTrue(due_t0["coordinator"])
        self.assertFalse(due_t1["coordinator"])
        self.assertTrue(due_t2["port"])
        self.assertEqual(cadence.message_latency_steps, 2)

    def test_multi_coordinator_directives_shape(self) -> None:
        cfg = get_default_config(num_ports=5, num_vessels=8, num_coordinators=2)
        rng = make_rng(42)
        vessels = initialize_vessels(
            num_vessels=cfg["num_vessels"],
            num_ports=cfg["num_ports"],
            nominal_speed=cfg["nominal_speed"],
            rng=rng,
        )
        medium = np.zeros((cfg["num_ports"], cfg["medium_horizon_days"]), dtype=float)

        groups = assign_vessels_to_coordinators(vessels, cfg["num_coordinators"])
        self.assertEqual(set(groups.keys()), {0, 1})
        self.assertEqual(sum(len(v) for v in groups.values()), cfg["num_vessels"])

        directives = build_multi_coordinator_directives(
            medium_forecast=medium,
            vessels=vessels,
            num_coordinators=cfg["num_coordinators"],
        )
        self.assertEqual(len(directives), cfg["num_coordinators"])
        for i, directive in enumerate(directives):
            self.assertEqual(directive["coordinator_id"], i)
            self.assertIn("dest_port", directive)
            self.assertIn("assigned_vessel_ids", directive)


if __name__ == "__main__":
    unittest.main()

