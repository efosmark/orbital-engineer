"""Utilities for identifying OpenCL devices and Linux DRM card mappings.

This module provides helpers to:

- Extract PCI bus-device-function (BDF) identifiers from vendor-specific
  OpenCL device metadata.
- Query OpenCL devices for any available PCI/topology extension info.
- Map PCI BDF identifiers to ``/sys/class/drm/card*`` indices on Linux.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast
import pyopencl as cl

def read_int(p: Path):
    try:
        return int(p.read_text().strip())
    except Exception:
        return None


def get_amd_drm_card(platform_id: int, device_id: int) -> int | None:
    platform = cl.get_platforms()[platform_id]
    device = platform.get_devices()[device_id]

    topo = cast(Any, device.get_info(cl.device_info.TOPOLOGY_AMD))

    bus = topo.bus & 0xff
    dev = topo.device & 0xff
    func = topo.function & 0xff

    # Match against the DRM cards in sysfs.
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        pci_path = (card / "device").resolve()

        # Example:
        #   /sys/devices/.../0000:65:00.0
        bdf = pci_path.name

        try:
            domain_bus, dev_func = bdf.rsplit(":", 1)
            pci_bus = int(domain_bus.rsplit(":", 1)[1], 16)

            pci_dev_str, pci_func_str = dev_func.split(".")
            pci_dev = int(pci_dev_str, 16)
            pci_func = int(pci_func_str, 16)
        except ValueError:
            continue
        if (pci_bus == bus and pci_dev == dev and pci_func == func):
            return int(card.name.removeprefix("card"))
    return None

@dataclass
class GPUStatus:
    id: tuple[int,int]
    name:str
    platform:str
    utilization: float|None
    temperature: float|None
    power_watts: float|None
    
    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(
            id=d['id'],
            name=d['name'],
            platform=d['platform'],
            utilization=d['utilization'],
            temperature=d['temperature'],
            power_watts=d['power_watts']
        )

class CLDeviceManager:
    
    def __init__(self, platform_id, device_id):
        self.platform_id = platform_id
        self.device_id = device_id
        platform = cl.get_platforms()[platform_id]
        devices = platform.get_devices()
        self._device = devices[device_id]
        self.opencl_platform_name = platform.name
        self.opencl_device_name = self._device.name
        self.drm_card_index = get_amd_drm_card(platform_id, device_id)

    def amd_paths(self):
        card = self.drm_card_index
        if card is None:
            print("Cannot find card")
            return None, None
        base = Path(f"/sys/class/drm/card{card}/device")
        # utilization
        util = base / "gpu_busy_percent"
        # hwmon temps/power
        hwmons = sorted((base / "hwmon").glob("hwmon*"))
        return util, hwmons

    def gpu_status(self):
        util_p, hwmons = self.amd_paths()
        util = read_int(util_p) if util_p is not None else 0.0 # percent
        temps = {}
        power_w = None
        if hwmons:
            h = hwmons[0]
            # map temp*_label -> temp*_input
            for lbl in h.glob("temp*_label"):
                name = lbl.read_text().strip()
                idx = lbl.name.split('_')[0]  # e.g., temp1
                val = read_int(h / f"{idx}_input")
                if val is not None:
                    temps[name] = val / 1000.0
            pw = read_int(h / "power1_average")
            if pw is not None:
                power_w = pw / 1e6
        
        avg_temp = None
        if temps:
            avg_temp = sum(temps.values()) / len(temps)
        return GPUStatus(
            (self.platform_id, self.device_id),
            self.opencl_platform_name,
            self.opencl_device_name,
            util,
            avg_temp,
            power_w
        )
