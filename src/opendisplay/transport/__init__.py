"""Transport layer: BLE and TCP/LAN command/response links."""

from .base import Transport
from .connection import BLEConnection
from .ip import TcpTransport

# Transport-neutral alias; BLEConnection stays importable under its own name.
BleTransport = BLEConnection

__all__ = [
    "Transport",
    "BLEConnection",
    "BleTransport",
    "TcpTransport",
]
