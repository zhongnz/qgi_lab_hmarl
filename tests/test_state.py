"""Unit tests for state.py — PortState, VesselState, initializers, make_rng."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.state import (
    PortState,
    VesselState,
    initialize_ports,
    initialize_vessels,
    make_rng,
)


class MakeRngTests(unittest.TestCase):
    """Tests for make_rng helper."""

    def test_default_seed_deterministic(self) -> None:
        """Two calls with default seed produce the same sequence."""
        r1 = make_rng()
        r2 = make_rng()
        self.assertEqual(r1.integers(0, 1000), r2.integers(0, 1000))

    def test_custom_seed(self) -> None:
        r1 = make_rng(123)
        r2 = make_rng(123)
        self.assertEqual(r1.integers(0, 1000), r2.integers(0, 1000))

    def test_different_seeds_differ(self) -> None:
        vals = [make_rng(s).integers(0, 2**31) for s in range(5)]
        self.assertEqual(len(set(vals)), 5)


class PortStateTests(unittest.TestCase):
    """Direct tests for PortState dataclass."""

    def test_defaults(self) -> None:
        p = PortState(port_id=0, queue=0, docks=3, occupied=0)
        self.assertEqual(p.cumulative_wait_hours, 0.0)
        self.assertEqual(p.vessels_served, 0)
        self.assertEqual(p.service_times, [])

    def test_service_times_independent(self) -> None:
        """Each instance gets its own service_times list."""
        p1 = PortState(port_id=0, queue=0, docks=2, occupied=0)
        p2 = PortState(port_id=1, queue=0, docks=2, occupied=0)
        p1.service_times.append(5.0)
        self.assertEqual(len(p2.service_times), 0)

    def test_fields_stored(self) -> None:
        p = PortState(
            port_id=3, queue=5, docks=4, occupied=2,
            service_times=[1.0, 2.0], cumulative_wait_hours=10.0,
            vessels_served=7,
        )
        self.assertEqual(p.port_id, 3)
        self.assertEqual(p.queue, 5)
        self.assertEqual(p.docks, 4)
        self.assertEqual(p.occupied, 2)
        self.assertEqual(p.service_times, [1.0, 2.0])
        self.assertAlmostEqual(p.cumulative_wait_hours, 10.0)
        self.assertEqual(p.vessels_served, 7)


class VesselStateTests(unittest.TestCase):
    """Direct tests for VesselState dataclass."""

    def test_defaults(self) -> None:
        v = VesselState(vessel_id=0, location=1, destination=2)
        self.assertAlmostEqual(v.position_nm, 0.0)
        self.assertAlmostEqual(v.speed, 12.0)
        self.assertAlmostEqual(v.fuel, 100.0)
        self.assertAlmostEqual(v.initial_fuel, 100.0)
        self.assertAlmostEqual(v.emissions, 0.0)
        self.assertAlmostEqual(v.delay_hours, 0.0)
        self.assertFalse(v.at_sea)

    def test_custom_values(self) -> None:
        v = VesselState(
            vessel_id=5, location=3, destination=4,
            position_nm=100.0, speed=15.0, fuel=80.0,
            initial_fuel=100.0, emissions=5.0,
            delay_hours=2.5, at_sea=True,
        )
        self.assertEqual(v.vessel_id, 5)
        self.assertTrue(v.at_sea)
        self.assertAlmostEqual(v.position_nm, 100.0)

    def test_mutability(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=1)
        v.fuel -= 10.0
        self.assertAlmostEqual(v.fuel, 90.0)


class InitializePortsTests(unittest.TestCase):
    """Tests for initialize_ports."""

    def test_correct_count(self) -> None:
        rng = make_rng(42)
        ports = initialize_ports(5, docks_per_port=3, rng=rng)
        self.assertEqual(len(ports), 5)

    def test_port_ids_sequential(self) -> None:
        rng = make_rng(0)
        ports = initialize_ports(4, docks_per_port=2, rng=rng)
        self.assertEqual([p.port_id for p in ports], [0, 1, 2, 3])

    def test_occupied_within_docks(self) -> None:
        rng = make_rng(99)
        ports = initialize_ports(10, docks_per_port=5, rng=rng)
        for p in ports:
            self.assertGreaterEqual(p.occupied, 0)
            self.assertLess(p.occupied, p.docks)

    def test_service_times_length_matches_occupied(self) -> None:
        rng = make_rng(7)
        ports = initialize_ports(8, docks_per_port=4, rng=rng)
        for p in ports:
            self.assertEqual(len(p.service_times), p.occupied)

    def test_custom_service_time(self) -> None:
        rng = make_rng(42)
        ports = initialize_ports(3, docks_per_port=2, rng=rng, service_time_hours=12.0)
        for p in ports:
            for st in p.service_times:
                self.assertAlmostEqual(st, 12.0)

    def test_queue_non_negative(self) -> None:
        rng = make_rng(42)
        ports = initialize_ports(6, docks_per_port=3, rng=rng)
        for p in ports:
            self.assertGreaterEqual(p.queue, 0)

    def test_single_port(self) -> None:
        rng = make_rng(0)
        ports = initialize_ports(1, docks_per_port=1, rng=rng)
        self.assertEqual(len(ports), 1)
        self.assertEqual(ports[0].port_id, 0)

    def test_zero_ports(self) -> None:
        rng = make_rng(0)
        ports = initialize_ports(0, docks_per_port=3, rng=rng)
        self.assertEqual(len(ports), 0)


class InitializeVesselsTests(unittest.TestCase):
    """Tests for initialize_vessels."""

    def test_correct_count(self) -> None:
        rng = make_rng(42)
        vessels = initialize_vessels(10, num_ports=5, nominal_speed=12.0, rng=rng)
        self.assertEqual(len(vessels), 10)

    def test_vessel_ids_sequential(self) -> None:
        rng = make_rng(0)
        vessels = initialize_vessels(4, num_ports=3, nominal_speed=12.0, rng=rng)
        self.assertEqual([v.vessel_id for v in vessels], [0, 1, 2, 3])

    def test_locations_in_range(self) -> None:
        rng = make_rng(99)
        vessels = initialize_vessels(20, num_ports=5, nominal_speed=12.0, rng=rng)
        for v in vessels:
            self.assertGreaterEqual(v.location, 0)
            self.assertLess(v.location, 5)

    def test_destinations_in_range(self) -> None:
        rng = make_rng(99)
        vessels = initialize_vessels(20, num_ports=5, nominal_speed=12.0, rng=rng)
        for v in vessels:
            self.assertGreaterEqual(v.destination, 0)
            self.assertLess(v.destination, 5)

    def test_speed_matches_nominal(self) -> None:
        rng = make_rng(42)
        vessels = initialize_vessels(5, num_ports=3, nominal_speed=15.0, rng=rng)
        for v in vessels:
            self.assertAlmostEqual(v.speed, 15.0)

    def test_fuel_matches_initial(self) -> None:
        rng = make_rng(42)
        vessels = initialize_vessels(5, num_ports=3, nominal_speed=12.0, rng=rng, initial_fuel=200.0)
        for v in vessels:
            self.assertAlmostEqual(v.fuel, 200.0)
            self.assertAlmostEqual(v.initial_fuel, 200.0)

    def test_not_at_sea_initially(self) -> None:
        rng = make_rng(42)
        vessels = initialize_vessels(5, num_ports=3, nominal_speed=12.0, rng=rng)
        for v in vessels:
            self.assertFalse(v.at_sea)

    def test_zero_vessels(self) -> None:
        rng = make_rng(0)
        vessels = initialize_vessels(0, num_ports=3, nominal_speed=12.0, rng=rng)
        self.assertEqual(len(vessels), 0)

    def test_deterministic_with_same_seed(self) -> None:
        v1 = initialize_vessels(5, num_ports=3, nominal_speed=12.0, rng=make_rng(42))
        v2 = initialize_vessels(5, num_ports=3, nominal_speed=12.0, rng=make_rng(42))
        for a, b in zip(v1, v2):
            self.assertEqual(a.location, b.location)
            self.assertEqual(a.destination, b.destination)


class BufferSetRewardDoneTests(unittest.TestCase):
    """Tests for the new set_reward / set_done public API on RolloutBuffer."""

    def test_set_reward_positive_index(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        for i in range(3):
            buf.add(obs=np.zeros(2), action=0.0, reward=float(i), done=False)
        buf.set_reward(1, 99.0)
        self.assertAlmostEqual(buf._rewards[1], 99.0)

    def test_set_reward_negative_index(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        for i in range(3):
            buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)
        buf.set_reward(-1, 42.0)
        self.assertAlmostEqual(buf._rewards[2], 42.0)

    def test_set_done_negative_index(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)
        buf.set_done(-1, 1.0)
        self.assertAlmostEqual(buf._dones[0], 1.0)

    def test_set_reward_out_of_range_raises(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)
        with self.assertRaises(IndexError):
            buf.set_reward(5, 1.0)

    def test_set_done_out_of_range_raises(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        with self.assertRaises(IndexError):
            buf.set_done(0, 1.0)

    def test_set_reward_empty_buffer_raises(self) -> None:
        from hmarl_mvp.buffer import RolloutBuffer
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        with self.assertRaises(IndexError):
            buf.set_reward(-1, 1.0)


if __name__ == "__main__":
    unittest.main()
