"""Unit tests for the PIPE_WRITE (0x0080-0x0082) sliding-window protocol.

Covers builders, response parsers, unpack_ack_ranges, classify_pipe_frame, and
the negotiation / probe-fallback layer. The sender loop (selective repeat, loss
recovery, PTO/RETX aborts, encryption, drain_stale) lives in
test_pipe_write_sender.py.
"""

from __future__ import annotations

import struct

import pytest
from epaper_dithering import ColorScheme

from opendisplay import OpenDisplayDevice
from opendisplay.exceptions import BLETimeoutError, InvalidResponseError
from opendisplay.models.capabilities import DeviceCapabilities
from opendisplay.models.config import (
    DisplayConfig,
    GlobalConfig,
    ManufacturerData,
    PowerOption,
    SystemConfig,
)
from opendisplay.models.enums import RefreshMode
from opendisplay.protocol import (
    DEFAULT_MAX_FRAME,
    PIPE_FLAG_COMPRESSED,
    PIPE_FRAME_OVERHEAD,
    PIPE_VERSION,
    CommandCode,
    PipeParams,
    build_pipe_write_data_command,
    build_pipe_write_end_command,
    build_pipe_write_start_command,
    classify_pipe_frame,
    parse_pipe_data_ack,
    parse_pipe_data_nack,
    parse_pipe_start_response,
    unpack_ack_ranges,
)
from opendisplay.protocol.responses import (
    PIPE_FRAME_ACK,
    PIPE_FRAME_END_ACK,
    PIPE_FRAME_END_NACK,
    PIPE_FRAME_NACK,
    PIPE_FRAME_OTHER,
)

# ─── Wire-frame helpers (shared with the sender tests via import) ─────────────


def start_ack(ver: int = 1, dw: int = 32, da: int = 32, df: int = 244, flags: int = 0x01) -> bytes:
    """Build a 0x0080 START ACK frame."""
    return b"\x00\x80" + bytes([ver, dw, da]) + struct.pack("<H", df) + bytes([flags])


def start_nack(err: int) -> bytes:
    """Build a 0x0080 START NACK frame."""
    return b"\xff\x80" + bytes([err, 0x00])


def data_ack(received: set[int]) -> bytes:
    """Build a 0x0081 DATA ACK reflecting a set of received chunk indexes.

    Uses cumulative highest_seen = max(received) plus a selective 32-bit mask.
    Only valid for indexes < 256 (unit tests never exceed that within one window).
    """
    hs = max(received)
    mask = 0
    for i in range(32):
        idx = hs - 1 - i
        if idx in received:
            mask |= 1 << i
    return b"\x00\x81" + bytes([hs % 256]) + struct.pack("<I", mask)


def data_ack_raw(highest_seen: int, mask: int) -> bytes:
    return b"\x00\x81" + bytes([highest_seen % 256]) + struct.pack("<I", mask)


def data_nack(err: int, highest_seen: int, mask: int) -> bytes:
    return b"\xff\x81" + bytes([err, highest_seen % 256]) + struct.pack("<I", mask)


END_ACK = b"\x00\x82"
END_NACK = b"\xff\x82"
REFRESH_COMPLETE = b"\x00\x73"
REFRESH_TIMEOUT = b"\x00\x74"


class ScriptedConn:
    """Fake BLE connection replaying scripted responses.

    Response items are either bytes (returned) or an Exception class/instance
    (raised — e.g. BLETimeoutError to simulate silence). Records written frames,
    their ``response`` and ``drain_stale`` flags, and a snapshot of the write
    count at each read (for window-invariant assertions).
    """

    def __init__(self, responses: list) -> None:
        self.written: list[bytes] = []
        self.write_responses: list[bool] = []
        self.drain_flags: list[bool] = []
        self._responses = list(responses)
        self.timeouts: list[float] = []
        self.writes_at_read: list[int] = []

    async def write_command(self, data: bytes, response: bool = True, drain_stale: bool = True) -> None:
        self.written.append(data)
        self.write_responses.append(response)
        self.drain_flags.append(drain_stale)

    async def read_response(self, timeout: float) -> bytes:
        self.timeouts.append(timeout)
        self.writes_at_read.append(len(self.written))
        if not self._responses:
            raise RuntimeError("ScriptedConn: no responses left")
        item = self._responses.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted")
        if isinstance(item, BaseException):
            raise item
        return item


def make_config(transmission_modes: int = 0x02) -> GlobalConfig:
    return GlobalConfig(
        system=SystemConfig(ic_type=0, communication_modes=0, device_flags=0, pwr_pin=0xFF, reserved=b"\x00" * 17),
        manufacturer=ManufacturerData(manufacturer_id=0, board_type=0, board_revision=0, reserved=b"\x00" * 18),
        power=PowerOption(
            power_mode=0,
            battery_capacity_mah=b"\x00\x00\x00",
            sleep_timeout_ms=0,
            tx_power=0,
            sleep_flags=0,
            battery_sense_pin=0xFF,
            battery_sense_enable_pin=0xFF,
            battery_sense_flags=0,
            capacity_estimator=0,
            voltage_scaling_factor=0,
            deep_sleep_current_ua=0,
            deep_sleep_time_seconds=0,
            reserved=b"\x00" * 12,
        ),
        displays=[
            DisplayConfig(
                instance_number=0,
                display_technology=0,
                panel_ic_type=0,
                pixel_width=8,
                pixel_height=8,
                active_width_mm=10,
                active_height_mm=10,
                tag_type=0,
                rotation=0,
                reset_pin=0xFF,
                busy_pin=0xFF,
                dc_pin=0xFF,
                cs_pin=0xFF,
                data_pin=0,
                partial_update_support=0,
                color_scheme=ColorScheme.MONO.value,
                transmission_modes=transmission_modes,
                clk_pin=0,
                reserved_pins=b"\x00" * 7,
                full_update_mC=0,
                reserved=b"\x00" * 13,
            )
        ],
    )


def make_device(
    responses: list,
    *,
    blocks_per_ack: int = 8,
    max_queue_size: int = 16,
    transmission_modes: int = 0x02,
) -> tuple[OpenDisplayDevice, ScriptedConn]:
    dev = OpenDisplayDevice(
        mac_address="AA:BB:CC:DD:EE:FF",
        config=make_config(transmission_modes),
        capabilities=DeviceCapabilities(width=8, height=8, color_scheme=ColorScheme.MONO),
        blocks_per_ack=blocks_per_ack,
        max_queue_size=max_queue_size,
    )
    conn = ScriptedConn(responses)
    dev._connection = conn
    return dev, conn


# ─── Builders ────────────────────────────────────────────────────────────────


def test_build_pipe_write_start_compressed() -> None:
    cmd = build_pipe_write_start_command(True, 16, 8, 244, 0x11223344)
    assert cmd[:2] == b"\x00\x80"
    assert cmd[2] == PIPE_VERSION
    assert cmd[3] == PIPE_FLAG_COMPRESSED
    assert cmd[4] == 16  # req_window
    assert cmd[5] == 8  # req_ack_every
    assert cmd[6:8] == struct.pack("<H", 244)
    assert cmd[8:12] == struct.pack("<I", 0x11223344)
    assert len(cmd) == 12


def test_build_pipe_write_start_uncompressed_flag_clear() -> None:
    cmd = build_pipe_write_start_command(False, 1, 1, 244, 0)
    assert cmd[3] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 256},
        {"ack_every": -1},
        {"max_frame": 0x10000},
        {"total_size": 0x1_0000_0000},
    ],
)
def test_build_pipe_write_start_range_validation(kwargs: dict) -> None:
    base = {"compressed": False, "window": 8, "ack_every": 4, "max_frame": 244, "total_size": 0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_pipe_write_start_command(**base)  # type: ignore[arg-type]


def test_build_pipe_write_data() -> None:
    cmd = build_pipe_write_data_command(200, b"payload")
    assert cmd == b"\x00\x81" + bytes([200]) + b"payload"


def test_build_pipe_write_data_seq_range() -> None:
    with pytest.raises(ValueError):
        build_pipe_write_data_command(256, b"x")


def test_build_pipe_write_end_no_etag() -> None:
    assert build_pipe_write_end_command(0) == b"\x00\x82\x00"


def test_build_pipe_write_end_with_etag_mirrors_direct_write() -> None:
    cmd = build_pipe_write_end_command(1, 0xDEADBEEF)
    assert cmd == b"\x00\x82" + bytes([1]) + (0xDEADBEEF).to_bytes(4, "big")


def test_build_pipe_write_end_etag_range() -> None:
    with pytest.raises(ValueError):
        build_pipe_write_end_command(0, 0x1_0000_0000)


# ─── parse_pipe_start_response ───────────────────────────────────────────────


def test_parse_start_ack() -> None:
    ok, payload = parse_pipe_start_response(start_ack(1, 32, 16, 244, 0x01))
    assert ok is True
    assert payload == (1, 32, 16, 244, 0x01)


def test_parse_start_ack_tolerates_trailing_bytes() -> None:
    ok, payload = parse_pipe_start_response(start_ack() + b"\xaa\xbb")
    assert ok is True
    assert payload[3] == 244  # type: ignore[index]


def test_parse_start_ack_too_short_raises() -> None:
    with pytest.raises(InvalidResponseError):
        parse_pipe_start_response(b"\x00\x80\x01\x20")  # only 4 bytes


@pytest.mark.parametrize("err", [0x01, 0x02, 0x03, 0x04])
def test_parse_start_nack(err: int) -> None:
    ok, payload = parse_pipe_start_response(start_nack(err))
    assert ok is False
    assert payload == err


def test_parse_start_bad_echo_raises() -> None:
    with pytest.raises(InvalidResponseError):
        parse_pipe_start_response(b"\x00\x70\x00\x00\x00\x00\x00\x00")


# ─── parse_pipe_data_ack / nack ──────────────────────────────────────────────


def test_parse_data_ack() -> None:
    hs, mask = parse_pipe_data_ack(data_ack_raw(9, 0xDEADBEEF))
    assert hs == 9
    assert mask == 0xDEADBEEF


def test_parse_data_ack_trailing_bytes() -> None:
    hs, mask = parse_pipe_data_ack(data_ack_raw(3, 0x7) + b"\xff")
    assert (hs, mask) == (3, 0x7)


def test_parse_data_ack_wrong_shape_raises() -> None:
    with pytest.raises(InvalidResponseError):
        parse_pipe_data_ack(b"\x00\x81\x03")  # too short


def test_parse_data_nack() -> None:
    err, hs, mask = parse_pipe_data_nack(data_nack(0x03, 5, 0x1F))
    assert (err, hs, mask) == (0x03, 5, 0x1F)


# ─── unpack_ack_ranges ───────────────────────────────────────────────────────


def test_unpack_contiguous() -> None:
    # received {0,1,2}, window_base 0
    assert unpack_ack_ranges(2, 0b11, 0) == {0, 1, 2}


def test_unpack_with_hole() -> None:
    # received {0,1,3,4}: hs=4 mask bits for 3,1,0
    mask = (1 << 0) | (1 << 2) | (1 << 3)  # 3, 1, 0
    got = unpack_ack_ranges(4, mask, 0)
    assert got == {0, 1, 3, 4}
    assert 2 not in got


def test_unpack_mod256_rollover() -> None:
    # window_base 250; receiver highest_seen wrapped to 4 (absolute 260)
    # received absolute 250..260 contiguous → mask all 1s for 10 below hs
    mask = (1 << 10) - 1
    got = unpack_ack_ranges(4, mask, 250)
    assert got == set(range(250, 261))


def test_unpack_stale_ack_below_window_base() -> None:
    # window_base already advanced to 10; a stale ACK reports highest_seen 9.
    got = unpack_ack_ranges(9, 0, 10)
    assert got == {9}  # resolves behind window_base, no spurious future index


# ─── classify_pipe_frame ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "frame,expected",
    [
        (data_ack_raw(3, 0), PIPE_FRAME_ACK),
        (data_nack(2, 3, 0), PIPE_FRAME_NACK),
        (END_ACK, PIPE_FRAME_END_ACK),
        (END_NACK, PIPE_FRAME_END_NACK),
        (b"\x00\x81\x03", PIPE_FRAME_OTHER),  # 3 bytes < 7 → not an ACK
        (b"\xff\x81\x02\x03\x00\x00\x00", PIPE_FRAME_OTHER),  # 7 bytes < 8 → not a NACK
        (b"\x00\x73", PIPE_FRAME_OTHER),
    ],
)
def test_classify_pipe_frame(frame: bytes, expected: str) -> None:
    assert classify_pipe_frame(frame) == expected


# ─── Negotiation / probe ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_negotiate_min_rule() -> None:
    dev, conn = make_device([start_ack(dw=32, da=32, df=244, flags=0x01)], blocks_per_ack=8, max_queue_size=16)
    params = await dev._negotiate_pipe(compressed=True, total_size=100)
    assert params == PipeParams(window=16, ack_every=8, max_frame=244, selective=True, compressed=True)
    # 0x0080 sent with the requested params.
    assert conn.written[0][:2] == b"\x00\x80"
    assert conn.timeouts[0] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_negotiate_window_clamped_to_32() -> None:
    dev, _ = make_device([start_ack(dw=64, da=64, df=244)], blocks_per_ack=64, max_queue_size=64)
    params = await dev._negotiate_pipe(compressed=False, total_size=10)
    assert params is not None
    assert params.window == 32  # hard cap
    assert params.ack_every == 32  # min(req, dev, W)


@pytest.mark.asyncio
async def test_negotiate_ack_every_clamped_to_window() -> None:
    dev, _ = make_device([start_ack(dw=4, da=32, df=244)], blocks_per_ack=32, max_queue_size=16)
    params = await dev._negotiate_pipe(compressed=False, total_size=10)
    assert params is not None
    assert params.window == 4
    assert params.ack_every == 4  # N clamped to W


@pytest.mark.asyncio
async def test_negotiate_floor_one() -> None:
    dev, _ = make_device([start_ack(dw=0, da=0, df=244)], blocks_per_ack=1, max_queue_size=1)
    # max_queue_size=1 disables pipe entirely, so call with a device that allows it:
    dev2, _ = make_device([start_ack(dw=0, da=0, df=244)], blocks_per_ack=1, max_queue_size=2)
    params = await dev2._negotiate_pipe(compressed=False, total_size=10)
    assert params is not None
    assert params.window == 1
    assert params.ack_every == 1


@pytest.mark.asyncio
async def test_negotiate_frame_min() -> None:
    dev, _ = make_device([start_ack(dw=32, da=32, df=180)], max_queue_size=16)
    params = await dev._negotiate_pipe(compressed=False, total_size=10)
    assert params is not None
    assert params.max_frame == min(DEFAULT_MAX_FRAME, 180)


@pytest.mark.asyncio
async def test_negotiate_silence_returns_none() -> None:
    dev, conn = make_device([BLETimeoutError], max_queue_size=16)
    params = await dev._negotiate_pipe(compressed=True, total_size=10)
    assert params is None


@pytest.mark.asyncio
async def test_negotiate_nack_bad_params_returns_none() -> None:
    dev, _ = make_device([start_nack(0x01)], max_queue_size=16)
    assert await dev._negotiate_pipe(compressed=True, total_size=10) is None


@pytest.mark.asyncio
async def test_negotiate_nack_compression_retries_uncompressed() -> None:
    dev, conn = make_device([start_nack(0x02), start_ack(flags=0x01)], max_queue_size=16)
    params = await dev._negotiate_pipe(compressed=True, total_size=10)
    assert params is not None
    assert params.compressed is False
    # Two 0x0080 writes: first compressed, second uncompressed (flags=0).
    starts = [w for w in conn.written if w[:2] == b"\x00\x80"]
    assert len(starts) == 2
    assert starts[0][3] == PIPE_FLAG_COMPRESSED
    assert starts[1][3] == 0


@pytest.mark.asyncio
async def test_negotiate_nack_compression_no_double_retry() -> None:
    # Second attempt also NACKs 0x02 → give up (no infinite recursion).
    dev, conn = make_device([start_nack(0x02), start_nack(0x02)], max_queue_size=16)
    assert await dev._negotiate_pipe(compressed=True, total_size=10) is None
    assert len([w for w in conn.written if w[:2] == b"\x00\x80"]) == 2


# ─── Routing / probe-cache via _execute_upload ───────────────────────────────


@pytest.mark.asyncio
async def test_max_queue_size_one_skips_probe() -> None:
    """max_queue_size <= 1 → no 0x0080 at all; straight to legacy uncompressed."""
    dev, conn = make_device(
        [b"\x00\x70", b"\x00\x71", b"\x00\x72", REFRESH_COMPLETE],
        max_queue_size=1,
    )
    await dev._execute_upload(b"ABCD", RefreshMode.FULL, use_compression=False)
    assert conn.written[0][:2] == b"\x00\x70"  # no probe first
    assert all(w[:2] != b"\x00\x80" for w in conn.written)


@pytest.mark.asyncio
async def test_silence_falls_back_to_legacy_and_caches() -> None:
    dev, conn = make_device(
        [
            BLETimeoutError,  # 0x0080 probe → silence
            b"\x00\x70",  # legacy START ACK
            b"\x00\x71",  # data ACK
            b"\x00\x72",  # END ACK
            REFRESH_COMPLETE,
        ],
        max_queue_size=16,
    )
    await dev._execute_upload(b"ABCD", RefreshMode.FULL, use_compression=False)
    assert conn.written[0][:2] == b"\x00\x80"  # probed once
    assert conn.written[1][:2] == b"\x00\x70"  # then legacy
    assert dev._pipe_probed is True
    assert dev._pipe_supported is False


@pytest.mark.asyncio
async def test_probe_cache_skips_second_probe() -> None:
    dev, conn = make_device(
        [
            BLETimeoutError,
            b"\x00\x70",
            b"\x00\x71",
            b"\x00\x72",
            REFRESH_COMPLETE,
            # second upload: legacy only, no probe
            b"\x00\x70",
            b"\x00\x71",
            b"\x00\x72",
            REFRESH_COMPLETE,
        ],
        max_queue_size=16,
    )
    await dev._execute_upload(b"ABCD", RefreshMode.FULL, use_compression=False)
    n_probes_1 = len([w for w in conn.written if w[:2] == b"\x00\x80"])
    await dev._execute_upload(b"EFGH", RefreshMode.FULL, use_compression=False)
    n_probes_2 = len([w for w in conn.written if w[:2] == b"\x00\x80"])
    assert n_probes_1 == 1
    assert n_probes_2 == 1  # no additional probe on the second upload


@pytest.mark.asyncio
async def test_disconnect_resets_pipe_cache() -> None:
    dev, _ = make_device([], max_queue_size=16)
    dev._pipe_probed = True
    dev._pipe_supported = False
    dev._pipe_params = PipeParams(8, 4, 244, True, False)
    dev._on_ble_disconnect()
    assert dev._pipe_probed is False
    assert dev._pipe_supported is False
    assert dev._pipe_params is None


def test_supports_pipe_write_bit() -> None:
    assert make_config(transmission_modes=0x10).displays[0].supports_pipe_write is True
    assert make_config(transmission_modes=0x02).displays[0].supports_pipe_write is False


def test_pipe_frame_overhead_constant() -> None:
    # Plaintext data capacity at 244: 244 - 3 = 241.
    assert DEFAULT_MAX_FRAME - PIPE_FRAME_OVERHEAD == 241


@pytest.mark.asyncio
async def test_execute_upload_uncompressed_pipe_auto_completes_end_to_end() -> None:
    """Uncompressed pipe upload through _execute_upload: firmware flush-ACKs then
    auto-completes with an unsolicited END_ACK — no explicit END is sent and the
    etag is reported as NOT committed (auto-complete stores no etag)."""
    dev, conn = make_device(
        [start_ack(), data_ack({0}), END_ACK, REFRESH_COMPLETE],
        max_queue_size=16,
    )
    committed = await dev._execute_upload(b"ABCD", RefreshMode.FULL, use_compression=False, new_etag=0x1234)
    assert committed is False  # etag never committed on auto-complete
    assert any(w[:2] == b"\x00\x81" for w in conn.written)  # data went via pipe frames
    assert all(w[:2] != b"\x00\x82" for w in conn.written)  # no spurious END
    assert all(w[:2] != b"\x00\x70" for w in conn.written)  # never fell back to legacy


@pytest.mark.asyncio
async def test_legacy_uncompressed_flow_byte_identical() -> None:
    """With pipe disabled, the legacy uncompressed wire bytes are unchanged."""
    dev, conn = make_device(
        [b"\x00\x70", b"\x00\x71", b"\x00\x72", REFRESH_COMPLETE],
        max_queue_size=1,
    )
    await dev._execute_upload(b"ABCD", RefreshMode.FULL, use_compression=False)
    assert conn.written == [
        b"\x00\x70",  # START (uncompressed)
        b"\x00\x71ABCD",  # single DATA chunk
        b"\x00\x72\x00",  # END, refresh mode 0
    ]
    # DATA chunk uses Write Without Response; START/END use Write Request.
    assert conn.write_responses == [True, False, True]
    # Legacy path drains stale before every write (default True).
    assert conn.drain_flags == [True, True, True]


def test_pipeline_chunks_removed() -> None:
    import opendisplay.protocol.commands as commands

    assert not hasattr(commands, "PIPELINE_CHUNKS")
    assert CommandCode.PIPE_WRITE_START == 0x0080
    assert CommandCode.PIPE_WRITE_DATA == 0x0081
    assert CommandCode.PIPE_WRITE_END == 0x0082
