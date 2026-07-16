"""Pure, dependency-light decoders for DEBUG instrumentation of BLE frames.

This module intentionally has NO dependency on :mod:`opendisplay.crypto` (or any
other package module) so it can be imported from the transport layer without
creating import cycles. It performs no I/O and raises no exceptions on short or
empty input — it is purely observational and safe to call from hot paths behind
an ``if _LOGGER.isEnabledFor(logging.DEBUG):`` guard.

Encrypted BLE frames use the layout::

    [cmd_echo:2][session_id:8][response_counter:8][ciphertext][tag:12]

where ``session_id`` || ``response_counter`` (big-endian) is the 16-byte full
nonce. The device increments ``response_counter`` once per emitted frame, so the
same counter observed twice signals a duplicate delivery, and a counter that
jumps past the current stream signals a second firmware stream.
"""

from __future__ import annotations

from typing import NamedTuple

# Minimum length of an encrypted frame: cmd(2) + nonce(16) + payload(1) + tag(12).
_ENCRYPTED_MIN_LEN = 31


class FrameInfo(NamedTuple):
    """Decoded view of a raw BLE frame for logging.

    Attributes:
        length: Total byte length of the raw frame.
        cmd_echo: The 2-byte big-endian command echo (0 if the frame is shorter
            than 2 bytes).
        is_encrypted: True if the frame is long enough to be an encrypted frame.
        session_id: Hex of the 8-byte session id (nonce_full[0:8]), or None if
            the frame is not encrypted.
        response_counter: The device→client counter (nonce_full[8:16], big-endian),
            or None if the frame is not encrypted.
    """

    length: int
    cmd_echo: int
    is_encrypted: bool
    session_id: str | None
    response_counter: int | None


def decode_frame(raw: bytes) -> FrameInfo:
    """Decode a raw BLE frame into a :class:`FrameInfo` without raising.

    Args:
        raw: Raw notification bytes (may be empty or short).

    Returns:
        A FrameInfo describing the frame. For frames shorter than the encrypted
        minimum, ``is_encrypted`` is False and ``session_id``/``response_counter``
        are None.
    """
    length = len(raw)
    cmd_echo = int.from_bytes(raw[0:2], "big") if length >= 2 else 0
    is_encrypted = length >= _ENCRYPTED_MIN_LEN
    if is_encrypted:
        session_id: str | None = raw[2:10].hex()
        response_counter: int | None = int.from_bytes(raw[10:18], "big")
    else:
        session_id = None
        response_counter = None
    return FrameInfo(
        length=length,
        cmd_echo=cmd_echo,
        is_encrypted=is_encrypted,
        session_id=session_id,
        response_counter=response_counter,
    )


def format_frame(raw: bytes) -> str:
    """Render a compact one-line summary of a raw BLE frame for DEBUG logging.

    Example::

        len=129 cmd=0x0040 enc=1 sid=ab12cd34ef56aa77 ctr=7 head=0040ab12… tail=…3f

    Args:
        raw: Raw notification bytes (may be empty or short).

    Returns:
        A single-line, human-readable summary. Never raises.
    """
    info = decode_frame(raw)
    sid = info.session_id if info.session_id is not None else "-"
    ctr = str(info.response_counter) if info.response_counter is not None else "-"
    head = raw[:4].hex()
    head_str = f"{head}…" if len(raw) > 4 else head
    tail_str = f"…{raw[-1:].hex()}" if len(raw) > 4 else "-"
    return (
        f"len={info.length} cmd=0x{info.cmd_echo:04x} enc={int(info.is_encrypted)} "
        f"sid={sid} ctr={ctr} head={head_str} tail={tail_str}"
    )
