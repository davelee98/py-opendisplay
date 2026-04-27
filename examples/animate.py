"""Animate a sequence of images on an OpenDisplay device.

Connects once, sends the first frame as a full update, then loops through
subsequent frames using partial (delta) updates. Repeats indefinitely.

Usage:
    python examples/animate.py --device AA:BB:CC:DD:EE:FF --interval 500 frame1.png frame2.png ...
    python examples/animate.py --device AA:BB:CC:DD:EE:FF --interval 1000 "frames/*.png"
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import sys
import time
from pathlib import Path
from collections.abc import Coroutine
from typing import Any, NoReturn, TypeVar

from epaper_dithering import DitherMode
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from opendisplay.device import OpenDisplayDevice, prepare_image
from opendisplay.exceptions import (
    AuthenticationFailedError,
    AuthenticationRequiredError,
    BLEConnectionError,
    BLETimeoutError,
    OpenDisplayError,
)
from opendisplay.models.enums import RefreshMode
from opendisplay.partial import PartialState

_T = TypeVar("_T")

_console = Console(stderr=True)

_DITHER_CHOICES: dict[str, DitherMode] = {m.name.lower().replace("_", "-"): m for m in DitherMode}


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def _error(msg: str) -> NoReturn:
    _console.print(f"[bold red]Error:[/bold red] {msg}")
    sys.exit(1)


def _handle_ble_error(exc: OpenDisplayError) -> NoReturn:
    if isinstance(exc, AuthenticationRequiredError):
        _error("Device requires an encryption key. Pass --key HEX.")
    if isinstance(exc, AuthenticationFailedError):
        _error("Authentication failed. Check that --key is correct.")
    if isinstance(exc, BLETimeoutError):
        _error(f"BLE timeout: {exc}")
    if isinstance(exc, BLEConnectionError):
        _error(f"BLE connection failed: {exc}")
    _error(f"Device error: {exc}")


def _parse_hex_key(hex_str: str | None) -> bytes | None:
    if hex_str is None:
        return None
    cleaned = hex_str.strip().replace(" ", "").replace(":", "")
    if len(cleaned) != 32:
        _error(f"--key must be exactly 32 hex characters (16 bytes), got {len(cleaned)}")
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        _error(f"--key contains invalid hex characters: {exc}")


def _device_kwargs(device: str, key: bytes | None, timeout: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": timeout, "encryption_key": key}
    if ":" in device or (len(device) == 36 and device.count("-") == 4):
        kwargs["mac_address"] = device
    else:
        kwargs["device_name"] = device
    return kwargs


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        force=True,
    )
    logging.getLogger("bleak").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)


def _expand_paths(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        expanded.extend(matches if matches else [p])
    return expanded


def _load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()
            images.append(img)
        except FileNotFoundError:
            _error(f"Image file not found: {p}")
        except UnidentifiedImageError:
            _error(f"Cannot open image (unsupported format): {p}")
    return images


async def _animate(
    device_kwargs: dict[str, Any],
    images: list[Image.Image],
    names: list[str],
    interval_ms: int,
    dither_mode: DitherMode,
) -> None:
    total = len(images)
    delay = interval_ms / 1000.0

    spinner_progress = Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("{task.description}"),
        console=_console,
    )
    bar_progress = Progress(
        BarColumn(),
        TaskProgressColumn(),
        console=_console,
    )

    class _Display:
        def __rich_console__(self, _con, _opts):  # type: ignore[no-untyped-def]
            yield spinner_progress
            if any(t.visible for t in bar_progress.tasks):
                yield bar_progress

    try:
        with Live(_Display(), console=_console, refresh_per_second=10, transient=False):
            status_task = spinner_progress.add_task("Connecting...", total=None)
            bar_task = bar_progress.add_task("", total=None, visible=False)

            async with OpenDisplayDevice(**device_kwargs) as device:
                spinner_progress.update(status_task, description=f"Pre-processing {total} frame(s)...")
                prep_times: list[float] = []
                prepared = []
                for img in images:
                    t0 = time.perf_counter()
                    prepared.append(
                        prepare_image(img, config=device.config, capabilities=device.capabilities, dither_mode=dither_mode)
                    )
                    prep_times.append(time.perf_counter() - t0)

                state = PartialState()
                frame_count = 0

                try:
                    while True:
                        idx = frame_count % total
                        is_first = frame_count == 0
                        refresh_mode = RefreshMode.FULL if is_first else RefreshMode.PARTIAL
                        update_type = "full" if is_first else "partial"
                        name = names[idx]

                        def _status(phase: str, _idx: int = idx, _name: str = name, _ut: str = update_type) -> str:
                            return f"{phase}  {_idx + 1}/{total}  [dim]{_name}[/dim]  ({_ut})"

                        send_end: list[float] = []
                        bytes_transferred: list[int] = []

                        def on_progress(sent: int, total_bytes: int) -> None:
                            bar_progress.update(bar_task, total=total_bytes, completed=sent, visible=True)
                            if sent == total_bytes:
                                send_end.append(time.perf_counter())
                                bytes_transferred.append(total_bytes)
                                bar_progress.update(bar_task, visible=False)
                                spinner_progress.update(status_task, description=_status("Refreshing..."))

                        t_upload_start = time.perf_counter()
                        spinner_progress.update(status_task, description=_status("Sending"))
                        await device.upload_prepared_image(
                            prepared[idx],
                            refresh_mode=refresh_mode,
                            state=state,
                            progress_callback=on_progress,
                        )
                        t_upload_end = time.perf_counter()

                        prep_ms = prep_times[idx] * 1000
                        send_ms = (send_end[0] - t_upload_start) * 1000
                        refresh_ms = (t_upload_end - send_end[0]) * 1000
                        kb = bytes_transferred[0] / 1024
                        stats = f"prep {prep_ms:.0f}ms · send {send_ms:.0f}ms ({kb:.1f} KB) · refresh {refresh_ms:.0f}ms"

                        _console.print(f"[dim]{idx + 1}/{total}  {name}  ({update_type})  {stats}[/dim]")
                        spinner_progress.update(status_task, description=_status("Showing"))
                        frame_count += 1
                        await asyncio.sleep(delay)

                except (KeyboardInterrupt, asyncio.CancelledError):
                    spinner_progress.update(status_task, description="Stopped.", total=1, completed=1)
                    return

    except OpenDisplayError as exc:
        _handle_ble_error(exc)


def _cmd_animate(args: argparse.Namespace) -> None:
    key = _parse_hex_key(args.key)
    paths = _expand_paths(args.images)

    if len(paths) < 2:
        _error(f"At least 2 images are required for animation, got {len(paths)}.")

    images = _load_images(paths)  # validate and read all files before connecting

    names = [Path(p).name for p in paths]
    listing = ", ".join(names) if len(names) <= 4 else f"{names[0]}, {names[1]}, ..., {names[-1]}"
    _console.print(f"Found {len(paths)} image(s): {listing}")

    _run(_animate(
        _device_kwargs(args.device, key, args.timeout),
        images,
        names,
        args.interval,
        _DITHER_CHOICES[args.dither_mode],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="animate",
        description="Animate a sequence of images on an OpenDisplay e-ink device.",
        epilog='Example: python examples/animate.py --device AA:BB:CC:DD:EE:FF --interval 500 "frames/*.png"',
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--device", required=True, metavar="ADDR", help="Device MAC address or name")
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        metavar="MS",
        help="Delay between frames in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--dither-mode",
        choices=list(_DITHER_CHOICES),
        default="burkes",
        help="Dithering algorithm (default: burkes)",
    )
    parser.add_argument("--key", default=None, metavar="HEX", help="Encryption key as 32 hex characters")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        metavar="SECS",
        help="BLE timeout in seconds (default: 10.0)",
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Image files to animate (in order)")

    args = parser.parse_args()
    _setup_logging(args.verbose)
    _cmd_animate(args)


if __name__ == "__main__":
    main()
