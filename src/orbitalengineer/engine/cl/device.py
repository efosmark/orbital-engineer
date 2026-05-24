"""Utilities for identifying OpenCL devices and Linux DRM card mappings.

This module provides helpers to:

- Extract PCI bus-device-function (BDF) identifiers from vendor-specific
  OpenCL device metadata.
- Query OpenCL devices for any available PCI/topology extension info.
- Map PCI BDF identifiers to ``/sys/class/drm/card*`` indices on Linux.
"""

from pathlib import Path
import re
from typing import Any
import pyopencl as cl


def print_platforms():
    for plat in cl.get_platforms():
        print("PLATFORM  : ")
        print("  name    : ", plat.name)
        print("  vendor  : ", plat.vendor)
        print("  version :", plat.version)
        for dev in plat.get_devices():
            print("   DEVICE  :", dev.name)
            print("     type  :", cl.device_type.to_string(dev.type))
        print()


class CLDeviceManager:
    
    def __init__(self, platform_id, device_id):
        platform = cl.get_platforms()[platform_id]
        devices = platform.get_devices()
        self._device = devices[device_id]
        self.opencl_platform_name = platform.name
        self.opencl_device_name = self._device.name
        self.opencl_pci_bdf = self.get_device_pci_bdf(self._device)
        self.drm_card_index = self.map_pci_bdf_to_drm_card(self.opencl_pci_bdf)

    def _extract_pci_bdf(self, value:Any) -> str|None:
        """Extract a PCI BDF string from a vendor-specific topology value.

        The function first searches ``value`` converted to text for a PCI
        address in ``dddd:bb:dd.f`` form. If that fails, it attempts to read
        ``domain``, ``bus``, ``device``, and ``function`` attributes from
        object-like values and formats them to the same string representation.

        :param value: Raw topology or PCI information returned by OpenCL.
        :type value: Any
        :returns: Normalized lowercase PCI BDF string, or ``None`` when not found.
        :rtype: str | None
        """
        # Try direct BDF text first, then unpack known object layouts.
        text = str(value)
        match = re.search(r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])", text)
        if match: return match.group(1).lower()

        attrs = ("domain", "bus", "device", "function")
        if all(hasattr(value, a) for a in attrs):
            try:
                return (
                    f"{int(value.domain):04x}:"
                    f"{int(value.bus):02x}:"
                    f"{int(value.device):02x}."
                    f"{int(value.function)}"
                )
            except Exception:
                return None
        return None

    def get_device_pci_bdf(self, device:Any) -> str|None:
        """Query an OpenCL device for PCI BDF information.

        The function scans available ``pyopencl.device_info`` constants looking
        for entries related to PCI or topology extensions, then queries the
        device and tries to normalize the result into a PCI BDF string.

        :param device: OpenCL device object that supports ``get_info``.
        :type device: Any
        :returns: PCI BDF string in ``dddd:bb:dd.f`` format, or ``None`` when
        unavailable.
        :rtype: str | None
        """
        # Probe any available PCI/topology info constants across vendors/extensions.
        for name in dir(cl.device_info):
            upper = name.upper()
            if ("PCI" not in upper) and ("TOPOLOGY" not in upper):
                continue
            info = getattr(cl.device_info, name)
            try:
                value = device.get_info(info)
            except Exception:
                continue
            bdf = self._extract_pci_bdf(value)
            if bdf is not None:
                return bdf
        return None

    def map_pci_bdf_to_drm_card(self, pci_bdf:str|None) -> int|None:
        """Map a PCI BDF identifier to a Linux DRM card index.

        This inspects ``/sys/class/drm/card*`` entries and compares each
        ``device/uevent`` file against the provided PCI slot name.

        :param pci_bdf: PCI BDF string in ``dddd:bb:dd.f`` format.
        :type pci_bdf: str | None
        :returns: Numeric DRM card index (for example ``0`` for ``card0``), or
        ``None`` if no matching card is found.
        :rtype: int | None
        """
        if not pci_bdf:
            return None

        for card_path in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            uevent_path = card_path / "device" / "uevent"
            try:
                uevent = uevent_path.read_text()
            except Exception:
                continue
            if f"PCI_SLOT_NAME={pci_bdf}" in uevent:
                try:
                    return int(card_path.name.removeprefix("card"))
                except ValueError:
                    return None
        return None
