"""mDNS / DNS-SD discovery of WiFi (LAN) OpenDisplay devices.

Browses the ``_opendisplay._tcp.local.`` service (protocol 2.2 SECTION 9) and
reads the device TXT record (``mac`` REQUIRED, ``tls`` REQUIRED, plus optional
``fw`` / ``cm`` / ``msd`` / ``id`` / ``pv``). The ``mac`` value is passed through
**raw** (lowercase-colon from the TXT); Home Assistant uppercases it to match the
BlueZ form it stores.

Requires the optional ``zeroconf`` dependency (``pip install py-opendisplay[wifi]``).
The import is guarded so the core BLE package works without it installed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .protocol import OD_LAN_MDNS_SERVICE

_LOGGER = logging.getLogger(__name__)

# DNS-SD fully-qualified service type (the browse form is OD_LAN_MDNS_SERVICE).
_SERVICE_TYPE = f"{OD_LAN_MDNS_SERVICE}.local."

_MISSING_ZEROCONF_MSG = (
    "WiFi/LAN discovery requires the 'zeroconf' package. "
    "Install it with: pip install py-opendisplay[wifi]"
)


@dataclass(frozen=True)
class IpDeviceInfo:
    """A WiFi-reachable OpenDisplay device discovered via mDNS."""

    name: str
    host: str
    port: int
    mac: str | None  # raw lowercase-colon from the TXT record; None if absent
    tls: bool
    msd: str | None = None
    fw: str | None = None
    cm: str | None = None


def _txt_str(properties: dict[bytes, bytes | None], key: str) -> str | None:
    """Decode a TXT value to str, or None if absent/undecodable."""
    raw = properties.get(key.encode())
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _txt_bool(properties: dict[bytes, bytes | None], key: str) -> bool:
    """Interpret a TXT flag as boolean (``1`` / ``true`` / ``yes`` → True)."""
    value = _txt_str(properties, key)
    return value is not None and value.strip().lower() in ("1", "true", "yes")


async def discover_ip_devices(scan_seconds: float = 3.0) -> dict[str, IpDeviceInfo]:
    """Discover WiFi OpenDisplay devices advertised via mDNS.

    Args:
        scan_seconds: How long to browse for services before resolving them.

    Returns:
        Mapping of friendly device name -> :class:`IpDeviceInfo`.

    Raises:
        RuntimeError: If the optional ``zeroconf`` dependency is not installed.
    """
    try:
        from zeroconf import ServiceStateChange
        from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf
    except ImportError as err:  # pragma: no cover - exercised only without extra
        raise RuntimeError(_MISSING_ZEROCONF_MSG) from err

    found_names: set[str] = set()

    def _on_change(
        _zeroconf: object,
        _service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change is ServiceStateChange.Added:
            found_names.add(name)

    result: dict[str, IpDeviceInfo] = {}
    aiozc = AsyncZeroconf()
    try:
        browser = AsyncServiceBrowser(aiozc.zeroconf, _SERVICE_TYPE, handlers=[_on_change])
        await asyncio.sleep(scan_seconds)
        await browser.async_cancel()

        for service_name in found_names:
            info = AsyncServiceInfo(_SERVICE_TYPE, service_name)
            if not await info.async_request(aiozc.zeroconf, timeout=3000):
                _LOGGER.debug("mDNS service %s did not resolve", service_name)
                continue
            addresses = info.parsed_scoped_addresses()
            if not addresses or info.port is None:
                _LOGGER.debug("mDNS service %s has no address/port", service_name)
                continue
            properties: dict[bytes, bytes | None] = info.properties
            friendly = service_name.removesuffix("." + _SERVICE_TYPE)
            if friendly == service_name:  # unexpected suffix; fall back to first label
                friendly = service_name.split(".", 1)[0]
            device = IpDeviceInfo(
                name=friendly,
                host=addresses[0],
                port=info.port,
                mac=_txt_str(properties, "mac"),
                tls=_txt_bool(properties, "tls"),
                msd=_txt_str(properties, "msd"),
                fw=_txt_str(properties, "fw"),
                cm=_txt_str(properties, "cm"),
            )
            result[friendly] = device
            _LOGGER.debug("Discovered LAN device %s at %s:%d (tls=%s)", friendly, device.host, device.port, device.tls)
    finally:
        await aiozc.async_close()

    _LOGGER.info("LAN discovery complete: found %d device(s)", len(result))
    return result
