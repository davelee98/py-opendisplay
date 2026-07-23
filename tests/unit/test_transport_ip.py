"""Tests for the TCP/LAN transport framer and lifecycle (TcpTransport).

The [len:2 LE][payload] framer is exercised two ways: by injecting an
``asyncio.StreamReader`` (precise control over partial / coalesced / truncated
byte streams) and by a real loopback asyncio server (end-to-end open_connection
+ write + read).
"""

from __future__ import annotations

import asyncio

import pytest

from opendisplay.exceptions import OpenDisplayConnectionError, OpenDisplayTimeoutError
from opendisplay.protocol import OD_LAN_MAX_PAYLOAD
from opendisplay.transport.ip import TcpTransport


class _FakeWriter:
    """Minimal StreamWriter stand-in recording written bytes."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed


def _framed(payload: bytes) -> bytes:
    return len(payload).to_bytes(2, "little") + payload


def _wire(reader: asyncio.StreamReader) -> TcpTransport:
    t = TcpTransport("127.0.0.1", 2446)
    t._reader = reader
    t._writer = _FakeWriter()  # type: ignore[assignment]
    return t


# ── write framing ────────────────────────────────────────────────────────────


async def test_write_command_prepends_little_endian_length() -> None:
    t = _wire(asyncio.StreamReader())
    await t.write_command(b"\x00\x40hello")
    assert bytes(t._writer.buffer) == _framed(b"\x00\x40hello")  # type: ignore[union-attr]


async def test_write_command_rejects_empty_and_oversize() -> None:
    t = _wire(asyncio.StreamReader())
    with pytest.raises(ValueError):
        await t.write_command(b"")
    with pytest.raises(ValueError):
        await t.write_command(b"\x00" * (OD_LAN_MAX_PAYLOAD + 1))


# ── read framing ─────────────────────────────────────────────────────────────


async def test_read_single_frame() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_framed(b"\x00\x71"))
    t = _wire(reader)
    assert await t.read_response(timeout=1.0) == b"\x00\x71"


async def test_read_coalesced_frames() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_framed(b"AAA") + _framed(b"BBBB"))
    t = _wire(reader)
    assert await t.read_response(timeout=1.0) == b"AAA"
    assert await t.read_response(timeout=1.0) == b"BBBB"


async def test_read_partial_frame_reassembles() -> None:
    reader = asyncio.StreamReader()
    t = _wire(reader)
    payload = b"partial-body"
    frame = _framed(payload)

    async def dribble() -> None:
        for byte in frame:
            reader.feed_data(bytes([byte]))
            await asyncio.sleep(0)

    feeder = asyncio.create_task(dribble())
    assert await t.read_response(timeout=1.0) == payload
    await feeder


async def test_read_large_max_payload_frame() -> None:
    reader = asyncio.StreamReader()
    payload = (bytes(range(256)) * 16)[:OD_LAN_MAX_PAYLOAD]  # exactly 4094 bytes
    assert len(payload) == OD_LAN_MAX_PAYLOAD
    reader.feed_data(_framed(payload))
    t = _wire(reader)
    assert await t.read_response(timeout=1.0) == payload


async def test_zero_length_frame_is_protocol_error_and_disconnects() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((0).to_bytes(2, "little"))
    t = _wire(reader)
    with pytest.raises(OpenDisplayConnectionError):
        await t.read_response(timeout=1.0)
    assert not t.is_connected  # connection dropped on protocol violation


async def test_oversize_frame_is_protocol_error() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((OD_LAN_MAX_PAYLOAD + 1).to_bytes(2, "little"))
    t = _wire(reader)
    with pytest.raises(OpenDisplayConnectionError):
        await t.read_response(timeout=1.0)


async def test_read_timeout() -> None:
    reader = asyncio.StreamReader()  # never fed
    t = _wire(reader)
    with pytest.raises(OpenDisplayTimeoutError):
        await t.read_response(timeout=0.02)


async def test_truncated_body_raises_connection_error() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((10).to_bytes(2, "little") + b"abc")  # promises 10, sends 3
    reader.feed_eof()
    t = _wire(reader)
    with pytest.raises(OpenDisplayConnectionError):
        await t.read_response(timeout=1.0)


async def test_read_when_not_connected_raises() -> None:
    t = TcpTransport("127.0.0.1", 2446)
    with pytest.raises(OpenDisplayConnectionError):
        await t.read_response(timeout=0.1)


# ── end-to-end loopback ──────────────────────────────────────────────────────


async def test_roundtrip_against_real_server() -> None:
    received: list[bytes] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        header = await reader.readexactly(2)
        length = int.from_bytes(header, "little")
        body = await reader.readexactly(length)
        received.append(body)
        # Echo a distinct framed response.
        writer.write(_framed(b"\x00\x71"))
        await writer.drain()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        t = TcpTransport("127.0.0.1", port, timeout=2.0)
        await t.connect()
        assert t.is_connected
        await t.write_command(b"\x00\x40ping")
        assert await t.read_response(timeout=2.0) == b"\x00\x71"
        await t.disconnect()
        assert not t.is_connected

    assert received == [b"\x00\x40ping"]


async def test_connect_refused_raises_connection_error() -> None:
    # Port 1 is privileged and closed; connect must fail fast with a neutral error.
    t = TcpTransport("127.0.0.1", 1, timeout=1.0)
    with pytest.raises(OpenDisplayConnectionError):
        await t.connect()


def test_class_attributes() -> None:
    t = TcpTransport("h", 2446)
    assert t.max_frame == OD_LAN_MAX_PAYLOAD
    assert t.supports_write_without_response is False
    assert t.device_name is None
