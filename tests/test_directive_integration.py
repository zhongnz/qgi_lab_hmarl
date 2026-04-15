"""Integration test: coordinator directive → message bus → vessel observation.

Verifies that when a coordinator issues a directive with a dest_port,
the directive is delivered via the message bus after the configured
latency, and the vessel sees the correct directive.
"""

from __future__ import annotations

import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.env import MaritimeEnv


@pytest.fixture()
def env() -> MaritimeEnv:
    cfg = get_default_config(
        num_ports=3,
        num_vessels=4,
        rollout_steps=30,
        weather_enabled=False,
    )
    env = MaritimeEnv(config=cfg)
    env.reset()
    return env


class TestDirectiveDelivery:
    """Directive enqueue → deliver → vessel visibility."""

    def test_directive_delivered_after_latency(self, env: MaritimeEnv) -> None:
        """A directive enqueued at step 0 is visible after message_latency_steps."""
        latency = env.cadence.message_latency_steps
        vessel_id = 0
        directive = {"dest_port": 2, "departure_window_hours": 6, "emission_budget": 40.0}

        env.bus.enqueue_directive(
            deliver_step=env.t + latency,
            vessel_id=vessel_id,
            directive=directive,
        )

        # Before delivery step, the directive should NOT be in latest_directives
        assert env.bus.get_latest_directive(vessel_id) is None

        # Deliver pending messages
        env.bus.deliver_due(env.t + latency)

        # Now the directive should be visible
        delivered = env.bus.get_latest_directive(vessel_id)
        assert delivered is not None
        assert delivered["dest_port"] == 2
        assert delivered["emission_budget"] == 40.0

    def test_directive_updates_compliance_tracking(self, env: MaritimeEnv) -> None:
        """After a directive is delivered, compliance checks use dest_port."""
        vessel = env.vessels[0]
        vessel_id = vessel.vessel_id

        # Enqueue and deliver immediately
        directive = {"dest_port": 1, "departure_window_hours": 6, "emission_budget": 50.0}
        env.bus.enqueue_directive(deliver_step=env.t, vessel_id=vessel_id, directive=directive)
        env.bus.deliver_due(env.t)

        # If vessel destination matches directive, it's compliant
        vessel.destination = 1
        d = env.bus.get_latest_directive(vessel_id)
        assert d is not None
        assert vessel.destination == int(d["dest_port"])

        # If vessel destination differs, it's non-compliant
        vessel.destination = 0
        assert vessel.destination != int(d["dest_port"])

    def test_step_delivers_coordinator_directives(self, env: MaritimeEnv) -> None:
        """A full env.step with coordinator actions enqueues directives for vessels."""
        # Build a no-op action dict
        actions: dict[str, dict[int, dict]] = {"vessel": {}, "port": {}, "coordinator": {}}
        for i in range(env.num_vessels):
            actions["vessel"][i] = {
                "target_speed": float(env.cfg["nominal_speed"]),
                "request_arrival_slot": False,
                "requested_arrival_time": 0.0,
            }
        for i in range(env.num_ports):
            actions["port"][i] = {"service_rate": 1, "accept_requests": 0}
        # Coordinator sends vessels to port 2
        actions["coordinator"][0] = {
            "dest_port": 2,
            "per_vessel_dest": {},
            "departure_window_hours": int(env.cfg["coordinator_departure_window_options"][0]),
            "emission_budget": 50.0,
        }

        # Step env — this should enqueue directives
        env.step(actions)

        latency = env.cadence.message_latency_steps
        # After enough steps for delivery, directives should arrive
        for _ in range(latency + 1):
            env.step(actions)

        # At least one vessel should now have a directive with dest_port
        any_has_directive = False
        for v in env.vessels:
            d = env.bus.get_latest_directive(v.vessel_id)
            if d and "dest_port" in d:
                any_has_directive = True
                break
        assert any_has_directive, "No vessel received a coordinator directive after stepping"

    def test_multiple_directives_latest_wins(self, env: MaritimeEnv) -> None:
        """When multiple directives are delivered, the latest one is returned."""
        vessel_id = 0

        # First directive: port 0
        env.bus.enqueue_directive(deliver_step=0, vessel_id=vessel_id, directive={"dest_port": 0})
        env.bus.deliver_due(0)

        # Second directive: port 2
        env.bus.enqueue_directive(deliver_step=1, vessel_id=vessel_id, directive={"dest_port": 2})
        env.bus.deliver_due(1)

        d = env.bus.get_latest_directive(vessel_id)
        assert d is not None
        assert d["dest_port"] == 2, "Latest directive should override earlier ones"
