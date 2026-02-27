"""Asynchronous message bus for inter-agent communication."""

from __future__ import annotations

from typing import Any


class MessageBus:
    """
    Manages three asynchronous message queues with configurable latency:

    1. Coordinator → Vessel directives
    2. Vessel → Port arrival-slot requests
    3. Port → Vessel slot responses (accept/reject)

    Messages are enqueued with a delivery step and delivered when the
    simulation clock reaches that step.
    """

    def __init__(self, num_ports: int) -> None:
        self.num_ports = num_ports
        self._directive_queue: list[tuple[int, int, dict[str, Any]]] = []
        self._arrival_request_queue: list[tuple[int, int, int, float]] = []
        self._slot_response_queue: list[tuple[int, int, bool, int]] = []
        self._pending_port_requests: dict[int, list[int]] = {
            port_id: [] for port_id in range(num_ports)
        }
        # Parallel list storing the requested_arrival_time for each pending request.
        self._pending_port_arrival_times: dict[int, list[float]] = {
            port_id: [] for port_id in range(num_ports)
        }
        self._awaiting_slot_response: set[int] = set()
        self._latest_directive_by_vessel: dict[int, dict[str, Any]] = {}

    def reset(self, num_ports: int | None = None) -> None:
        """Clear all queues and pending state."""
        if num_ports is not None:
            self.num_ports = num_ports
        self._directive_queue = []
        self._arrival_request_queue = []
        self._slot_response_queue = []
        self._pending_port_requests = {
            port_id: [] for port_id in range(self.num_ports)
        }
        self._pending_port_arrival_times = {
            port_id: [] for port_id in range(self.num_ports)
        }
        self._awaiting_slot_response = set()
        self._latest_directive_by_vessel = {}

    # -- Enqueueing ----------------------------------------------------------

    def enqueue_directive(
        self, deliver_step: int, vessel_id: int, directive: dict[str, Any]
    ) -> None:
        """Queue a coordinator directive for delivery at *deliver_step*."""
        self._directive_queue.append((deliver_step, vessel_id, directive))

    def enqueue_arrival_request(
        self,
        deliver_step: int,
        vessel_id: int,
        destination: int,
        requested_arrival_time: float = 0.0,
    ) -> None:
        """Queue a vessel arrival-slot request for delivery at *deliver_step*.

        Parameters
        ----------
        requested_arrival_time:
            The absolute simulation step by which the vessel wants to arrive
            (t_arr in the proposal).  A value of 0.0 means no preference;
            ports use this to prioritise earlier-deadline vessels.
        """
        self._arrival_request_queue.append(
            (deliver_step, vessel_id, destination, requested_arrival_time)
        )

    def enqueue_slot_response(
        self, deliver_step: int, vessel_id: int, accepted: bool, port_id: int
    ) -> None:
        """Queue a port slot response for delivery at *deliver_step*."""
        self._slot_response_queue.append((deliver_step, vessel_id, accepted, port_id))

    # -- Awaiting tracking ---------------------------------------------------

    def is_awaiting(self, vessel_id: int) -> bool:
        """Return True if the vessel is waiting for a slot response."""
        return vessel_id in self._awaiting_slot_response

    def mark_awaiting(self, vessel_id: int) -> None:
        """Mark a vessel as waiting for a slot response."""
        self._awaiting_slot_response.add(vessel_id)

    def clear_awaiting(self, vessel_id: int) -> None:
        """Remove the awaiting mark for a vessel."""
        self._awaiting_slot_response.discard(vessel_id)

    # -- Accessors -----------------------------------------------------------

    @property
    def latest_directives(self) -> dict[int, dict[str, Any]]:
        """Mapping of vessel_id → latest delivered directive (read-only copy)."""
        return dict(self._latest_directive_by_vessel)

    def get_latest_directive(self, vessel_id: int) -> dict[str, Any] | None:
        """Return the latest delivered directive for *vessel_id*, or None."""
        return self._latest_directive_by_vessel.get(vessel_id)

    def get_pending_requests(self, port_id: int) -> list[int]:
        """Return list of vessel IDs with pending arrival requests at *port_id*."""
        return self._pending_port_requests.get(port_id, [])

    def clear_pending_requests(self, port_id: int) -> None:
        """Clear the pending request queue for *port_id*."""
        self._pending_port_requests[port_id] = []
        self._pending_port_arrival_times[port_id] = []

    def get_pending_requests_sorted(self, port_id: int) -> list[int]:
        """Return vessel IDs with pending requests at *port_id*, sorted by requested
        arrival time (earliest deadline first).  Vessels with ``requested_arrival_time
        == 0.0`` (no preference) are placed last.
        """
        vessel_ids = self._pending_port_requests.get(port_id, [])
        arrival_times = self._pending_port_arrival_times.get(port_id, [])
        if not vessel_ids:
            return []
        # Pair up, put 0.0 (no preference) behind real deadlines.
        paired = list(zip(arrival_times, vessel_ids))
        paired.sort(key=lambda x: (x[0] == 0.0, x[0]))
        return [vid for _, vid in paired]

    @property
    def queue_sizes(self) -> dict[str, int]:
        """Current size of each internal queue."""
        return {
            "directives": len(self._directive_queue),
            "arrival_requests": len(self._arrival_request_queue),
            "slot_responses": len(self._slot_response_queue),
        }

    @property
    def total_pending_requests(self) -> int:
        """Total pending arrival requests across all ports."""
        return sum(len(q) for q in self._pending_port_requests.values())

    # -- Delivery ------------------------------------------------------------

    def deliver_due(self, current_step: int) -> dict[int, dict[str, Any]]:
        """
        Deliver all messages whose delivery step has arrived.

        Returns a mapping of vessel_id to slot-response payload for
        vessels that received a port response this tick.
        """
        delivered_responses: dict[int, dict[str, Any]] = {}

        # Directives
        remaining: list[tuple[int, int, dict[str, Any]]] = []
        for deliver_step, vessel_id, directive in self._directive_queue:
            if deliver_step <= current_step:
                self._latest_directive_by_vessel[vessel_id] = directive
            else:
                remaining.append((deliver_step, vessel_id, directive))
        self._directive_queue = remaining

        # Arrival requests → pending port queues
        remaining_requests: list[tuple[int, int, int, float]] = []
        for deliver_step, vessel_id, destination, arr_time in self._arrival_request_queue:
            if deliver_step <= current_step:
                if 0 <= destination < self.num_ports:
                    self._pending_port_requests[destination].append(vessel_id)
                    self._pending_port_arrival_times[destination].append(arr_time)
            else:
                remaining_requests.append((deliver_step, vessel_id, destination, arr_time))
        self._arrival_request_queue = remaining_requests

        # Slot responses
        remaining_responses: list[tuple[int, int, bool, int]] = []
        for deliver_step, vessel_id, accepted, destination in self._slot_response_queue:
            if deliver_step <= current_step:
                delivered_responses[vessel_id] = {
                    "accepted": bool(accepted),
                    "dest_port": int(destination),
                }
            else:
                remaining_responses.append(
                    (deliver_step, vessel_id, accepted, destination)
                )
        self._slot_response_queue = remaining_responses

        return delivered_responses

    def peek(
        self, current_step: int
    ) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
        """
        Preview message state without mutation.

        Returns ``(latest_directive_by_vessel, pending_port_requests)`` as
        they would appear after delivery at *current_step*.
        """
        latest = dict(self._latest_directive_by_vessel)
        pending: dict[int, list[int]] = {
            pid: list(q) for pid, q in self._pending_port_requests.items()
        }

        for deliver_step, vessel_id, directive in self._directive_queue:
            if deliver_step <= current_step:
                latest[vessel_id] = directive

        for deliver_step, vessel_id, destination, _arr_time in self._arrival_request_queue:
            if deliver_step <= current_step and 0 <= destination < self.num_ports:
                pending[destination].append(vessel_id)

        return latest, pending
