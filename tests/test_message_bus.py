"""Tests for the MessageBus inter-agent communication layer."""

from __future__ import annotations

import unittest

from hmarl_mvp.message_bus import MessageBus


class MessageBusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(num_ports=3)

    def test_reset_clears_all_queues(self) -> None:
        self.bus.enqueue_directive(5, 0, {"dest_port": 1})
        self.bus.enqueue_arrival_request(5, 0, 1)
        self.bus.enqueue_slot_response(5, 0, True, 1)
        self.bus.mark_awaiting(0)
        self.bus.reset()
        self.assertEqual(self.bus.queue_sizes["directives"], 0)
        self.assertEqual(self.bus.queue_sizes["arrival_requests"], 0)
        self.assertEqual(self.bus.queue_sizes["slot_responses"], 0)
        self.assertFalse(self.bus.is_awaiting(0))
        self.assertIsNone(self.bus.get_latest_directive(0))

    def test_directive_delivery(self) -> None:
        self.bus.enqueue_directive(2, 0, {"dest_port": 1})
        self.bus.deliver_due(1)  # too early
        self.assertIsNone(self.bus.get_latest_directive(0))
        self.bus.deliver_due(2)  # on time
        self.assertEqual(self.bus.get_latest_directive(0), {"dest_port": 1})

    def test_arrival_request_delivery(self) -> None:
        self.bus.enqueue_arrival_request(3, 7, 2)
        self.bus.deliver_due(2)  # too early
        self.assertEqual(self.bus.get_pending_requests(2), [])
        self.bus.deliver_due(3)
        self.assertEqual(self.bus.get_pending_requests(2), [7])

    def test_slot_response_delivery(self) -> None:
        self.bus.enqueue_slot_response(4, 0, True, 1)
        responses = self.bus.deliver_due(3)
        self.assertEqual(responses, {})
        responses = self.bus.deliver_due(4)
        self.assertEqual(responses, {0: {"accepted": True, "dest_port": 1}})

    def test_awaiting_tracking(self) -> None:
        self.assertFalse(self.bus.is_awaiting(5))
        self.bus.mark_awaiting(5)
        self.assertTrue(self.bus.is_awaiting(5))
        self.bus.clear_awaiting(5)
        self.assertFalse(self.bus.is_awaiting(5))

    def test_clear_pending_requests(self) -> None:
        self.bus.enqueue_arrival_request(0, 1, 0)
        self.bus.deliver_due(0)
        self.assertEqual(self.bus.get_pending_requests(0), [1])
        self.bus.clear_pending_requests(0)
        self.assertEqual(self.bus.get_pending_requests(0), [])

    def test_out_of_range_destination_ignored(self) -> None:
        self.bus.enqueue_arrival_request(0, 0, 99)  # port 99 doesn't exist
        self.bus.deliver_due(0)
        self.assertEqual(self.bus.total_pending_requests, 0)

    def test_peek_does_not_mutate(self) -> None:
        self.bus.enqueue_directive(1, 0, {"dest_port": 2})
        self.bus.enqueue_arrival_request(1, 1, 0)
        latest, pending = self.bus.peek(1)
        self.assertEqual(latest[0], {"dest_port": 2})
        self.assertEqual(pending[0], [1])
        # Original state unchanged
        self.assertIsNone(self.bus.get_latest_directive(0))
        self.assertEqual(self.bus.get_pending_requests(0), [])

    def test_queue_sizes(self) -> None:
        self.bus.enqueue_directive(5, 0, {})
        self.bus.enqueue_directive(5, 1, {})
        self.bus.enqueue_arrival_request(5, 0, 0)
        self.bus.enqueue_slot_response(5, 0, True, 0)
        sizes = self.bus.queue_sizes
        self.assertEqual(sizes["directives"], 2)
        self.assertEqual(sizes["arrival_requests"], 1)
        self.assertEqual(sizes["slot_responses"], 1)

    def test_reset_with_new_port_count(self) -> None:
        self.bus.reset(num_ports=5)
        self.assertEqual(self.bus.num_ports, 5)
        self.assertEqual(len(self.bus.get_pending_requests(4)), 0)


if __name__ == "__main__":
    unittest.main()
