"""Transport protocol shared by BLE and TCP/LAN connections.

Defines the structural interface (:class:`Transport`) that
:class:`~opendisplay.transport.connection.BLEConnection` and
:class:`~opendisplay.transport.ip.TcpTransport` both satisfy, so
:class:`~opendisplay.device.OpenDisplayDevice` can drive either without
knowing which link is underneath. Method names deliberately match the
existing ``BLEConnection`` API (``write_command`` / ``read_response`` /
``drain_notifications``) so call sites stay unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Structural interface for a command/response link to a device.

    A transport frames a stop-and-wait command stream: write one command,
    read one response frame. It carries no protocol semantics (encryption,
    upload chunking, etc.) — those live in ``OpenDisplayDevice``.
    """

    #: Maximum application payload the link can carry in a single frame. BLE is
    #: capped at the HA GATT write ceiling (244); LAN allows up to 4094
    #: (a 4096-byte wire frame including the 2-byte length prefix).
    max_frame: int

    #: Human-readable device name, if known (BLE advertised name); None on LAN.
    device_name: str | None

    @property
    def supports_write_without_response(self) -> bool:
        """Whether a fire-and-forget (unacknowledged) write is available.

        True for BLE write-without-response; False for TCP (every write is
        reliable). Declared read-only so a property-backed implementation
        (BLEConnection) and a plain attribute (TcpTransport) both satisfy it.
        """

    @property
    def is_connected(self) -> bool:
        """Whether the link is currently up."""

    async def connect(self) -> None:
        """Establish the link. Raises on failure (transport-neutral errors)."""

    async def disconnect(self) -> None:
        """Tear the link down. Best-effort; must not raise on a dead link."""

    async def write_command(self, data: bytes, response: bool = True, drain_stale: bool = True) -> None:
        """Send one command frame.

        Args:
            data: Command bytes to send.
            response: BLE only — request an acknowledged write. Accepted and
                ignored by transports where every write is already reliable.
            drain_stale: BLE only — discard queued stale frames first. Accepted
                and ignored where there is no notification queue.
        """

    async def read_response(self, timeout: float = 5.0) -> bytes:
        """Read exactly one response frame, or raise on timeout."""

    def drain_notifications(self) -> int:
        """Discard any buffered frames; return how many were dropped."""

    async def clear_cache(self) -> bool:
        """Clear any per-device GATT cache.

        Returns True if a cache was cleared, False if the transport has no
        cache (e.g. TCP, which always returns False).
        """
