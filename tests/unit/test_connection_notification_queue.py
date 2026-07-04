"""Tests for notification-queue draining and timeout recovery (C6)."""

from __future__ import annotations

import asyncio

import pytest

from opendisplay.exceptions import BLETimeoutError
from opendisplay.transport.connection import BLEConnection


def test_drain_notifications_discards_all_queued() -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    conn._notification_queue.put_nowait(b"a")
    conn._notification_queue.put_nowait(b"b")

    assert conn.drain_notifications() == 2
    assert conn._notification_queue.empty()


def test_drain_notifications_empty_queue_is_noop() -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    assert conn.drain_notifications() == 0


@pytest.mark.asyncio
async def test_read_response_recovers_item_delivered_during_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If wait_for cancels queue.get() after an item was handed over, the item
    is recovered synchronously instead of being lost."""
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    conn._notification_queue.put_nowait(b"late-response")

    def fake_wait_for(coro: object, timeout: float) -> object:
        coro.close()  # type: ignore[attr-defined]  # avoid unawaited-coroutine warning
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    assert await conn.read_response(timeout=0.01) == b"late-response"


@pytest.mark.asyncio
async def test_read_response_times_out_when_queue_empty() -> None:
    conn = BLEConnection("AA:BB:CC:DD:EE:FF")
    with pytest.raises(BLETimeoutError):
        await conn.read_response(timeout=0.01)
