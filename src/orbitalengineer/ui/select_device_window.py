from typing import Any

import pyopencl as cl

from orbitalengineer.ui.gtk4 import Gio, Gtk


class SelectDeviceWindow(Gtk.Window):
    def __init__(
        self,
        parent: Gtk.Window | None = None,
        platform_id: int = 0,
        device_id: int | None = None,
    ):
        super().__init__(
            title="Select OpenCL Device",
            transient_for=parent,
            modal=True,
        )
        self.platform_id: int | None = None
        self.device_id: int | None = None
        self._platforms = cl.get_platforms()
        self._devices_by_platform: list[list[Any]] = []

        self.set_default_size(640, 520)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        content.set_margin_top(18)
        content.set_margin_bottom(18)
        content.set_margin_start(18)
        content.set_margin_end(18)
        self.set_child(content)

        grid = Gtk.Grid(column_spacing=12, row_spacing=12)
        content.append(grid)

        platform_label = Gtk.Label(label="Platform", halign=Gtk.Align.START)
        device_label = Gtk.Label(label="Device", halign=Gtk.Align.START)
        self.platform_dropdown = Gtk.DropDown()
        self.device_dropdown = Gtk.DropDown()
        self.platform_dropdown.set_hexpand(True)
        self.device_dropdown.set_hexpand(True)

        grid.attach(platform_label, 0, 0, 1, 1)
        grid.attach(self.platform_dropdown, 1, 0, 1, 1)
        grid.attach(device_label, 0, 1, 1, 1)
        grid.attach(self.device_dropdown, 1, 1, 1, 1)

        details_frame = Gtk.Frame(label="Compute details")
        details_frame.set_vexpand(True)
        content.append(details_frame)

        details_scroll = Gtk.ScrolledWindow()
        details_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        details_scroll.set_min_content_height(260)
        details_scroll.set_propagate_natural_height(False)
        details_scroll.set_vexpand(True)
        details_frame.set_child(details_scroll)

        details_grid = Gtk.Grid(column_spacing=16, row_spacing=6)
        details_grid.set_margin_top(12)
        details_grid.set_margin_bottom(12)
        details_grid.set_margin_start(12)
        details_grid.set_margin_end(12)
        details_scroll.set_child(details_grid)

        self._detail_labels: dict[str, Gtk.Label] = {}
        detail_rows = (
            ("compute_units", "Compute units"),
            ("work_group_size", "Max work group size"),
            ("work_item_dims", "Max work item dims"),
            ("work_item_sizes", "Max work item sizes"),
            ("local_mem", "Local memory"),
            ("global_mem", "Global memory"),
            ("mem_alloc", "Max memory allocation"),
            ("constant_buffer", "Max constant buffer"),
            ("clock", "Max clock"),
            ("vector_width", "Preferred vector width"),
            ("native_vector_width", "Native vector width"),
            ("address_bits", "Address bits"),
            ("opencl_c", "OpenCL C"),
        )
        for row, (key, label_text) in enumerate(detail_rows):
            label = Gtk.Label(label=label_text, halign=Gtk.Align.START)
            value = Gtk.Label(label="-", halign=Gtk.Align.START)
            value.set_selectable(True)
            value.set_hexpand(True)
            details_grid.attach(label, 0, row, 1, 1)
            details_grid.attach(value, 1, row, 1, 1)
            self._detail_labels[key] = value

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        buttons.set_halign(Gtk.Align.END)
        content.append(buttons)

        cancel_button = Gtk.Button(label="Cancel")
        self.select_button = Gtk.Button(label="Select")
        self.select_button.add_css_class("suggested-action")
        buttons.append(cancel_button)
        buttons.append(self.select_button)

        self._load_platforms()
        self.platform_dropdown.set_sensitive(len(self._platforms) > 1)
        self.platform_dropdown.connect("notify::selected", self._on_platform_changed)
        self.device_dropdown.connect("notify::selected", self._on_device_changed)
        cancel_button.connect("clicked", self._on_cancel_clicked)
        self.select_button.connect("clicked", self._on_select_clicked)

        if self._platforms:
            platform_id = min(max(platform_id, 0), len(self._platforms) - 1)
            self.platform_dropdown.set_selected(platform_id)
            self._load_devices(platform_id, device_id)
        else:
            self.platform_dropdown.set_sensitive(False)
            self.device_dropdown.set_sensitive(False)
            self.select_button.set_sensitive(False)
            self._clear_device_details()

    def _load_platforms(self) -> None:
        labels: list[str] = []
        for platform in self._platforms:
            labels.append(self._platform_label(platform))
            try:
                devices = list(platform.get_devices())
            except cl.Error:
                devices = []
            self._devices_by_platform.append(devices)
        self.platform_dropdown.set_model(Gtk.StringList.new(labels))

    def _load_devices(self, platform_id: int, device_id: int | None = None) -> None:
        devices = self._devices_by_platform[platform_id]
        self.device_dropdown.set_model(
            Gtk.StringList.new([self._device_label(device) for device in devices])
        )

        has_devices = bool(devices)
        self.device_dropdown.set_sensitive(has_devices)
        self.select_button.set_sensitive(has_devices)
        if has_devices:
            if device_id is None:
                device_id = self._default_device_id(devices)
            device_id = min(max(device_id, 0), len(devices) - 1)
            self.device_dropdown.set_selected(device_id)
            self._update_device_details(devices[device_id])
        else:
            self._clear_device_details()

    def _on_platform_changed(self, dropdown: Gtk.DropDown, _param: Any) -> None:
        platform_id = dropdown.get_selected()
        if platform_id < len(self._platforms):
            self._load_devices(platform_id)

    def _on_device_changed(self, dropdown: Gtk.DropDown, _param: Any) -> None:
        platform_id = self.platform_dropdown.get_selected()
        device_id = dropdown.get_selected()
        if platform_id >= len(self._devices_by_platform):
            self._clear_device_details()
            return

        devices = self._devices_by_platform[platform_id]
        if device_id < len(devices):
            self._update_device_details(devices[device_id])
        else:
            self._clear_device_details()

    def _on_select_clicked(self, _button: Gtk.Button) -> None:
        platform_id = self.platform_dropdown.get_selected()
        device_id = self.device_dropdown.get_selected()
        if platform_id < len(self._platforms) and device_id < len(self._devices_by_platform[platform_id]):
            self.platform_id = platform_id
            self.device_id = device_id
        self.close()

    def _on_cancel_clicked(self, _button: Gtk.Button) -> None:
        self.platform_id = None
        self.device_id = None
        self.close()

    def get_selection(self) -> tuple[int, int] | None:
        if self.platform_id is None or self.device_id is None:
            return None
        return self.platform_id, self.device_id

    def _platform_label(self, platform: Any) -> str:
        vendor = getattr(platform, "vendor", "").strip()
        name = getattr(platform, "name", "Unknown platform").strip()
        return f"{name} ({vendor})" if vendor else name

    def _device_label(self, device: Any) -> str:
        name = getattr(device, "name", "Unknown device").strip()
        board_name = self._device_board_name(device)
        try:
            device_type = cl.device_type.to_string(device.type)
        except Exception:
            device_type = ""

        label = name
        if board_name and board_name != name:
            label = f"{board_name} - {name}"
        return f"{label} ({device_type})" if device_type else label

    def _device_board_name(self, device: Any) -> str:
        try:
            board_name_info = cl.device_info.BOARD_NAME_AMD
        except AttributeError:
            return ""

        try:
            value = device.get_info(board_name_info)
        except cl.Error:
            return ""
        return str(value).strip()

    def _update_device_details(self, device: Any) -> None:
        details = {
            "compute_units": self._device_attr(device, "max_compute_units"),
            "work_group_size": self._device_attr(device, "max_work_group_size"),
            "work_item_dims": self._device_attr(device, "max_work_item_dimensions"),
            "work_item_sizes": self._format_sequence(self._device_attr(device, "max_work_item_sizes")),
            "local_mem": self._format_bytes(self._device_attr(device, "local_mem_size")),
            "global_mem": self._format_bytes(self._device_attr(device, "global_mem_size")),
            "mem_alloc": self._format_bytes(self._device_attr(device, "max_mem_alloc_size")),
            "constant_buffer": self._format_bytes(self._device_attr(device, "max_constant_buffer_size")),
            "clock": self._format_mhz(self._device_attr(device, "max_clock_frequency")),
            "vector_width": self._format_vector_width(device, "preferred_vector_width"),
            "native_vector_width": self._format_vector_width(device, "native_vector_width"),
            "address_bits": self._device_attr(device, "address_bits"),
            "opencl_c": self._device_attr(device, "opencl_c_version"),
        }
        for key, value in details.items():
            self._detail_labels[key].set_label(self._format_value(value))

    def _device_attr(self, device: Any, name: str) -> Any:
        try:
            return getattr(device, name)
        except Exception:
            return None

    def _default_device_id(self, devices: list[Any]) -> int:
        return max(
            range(len(devices)),
            key=lambda index: self._device_compute_units(devices[index]),
        )

    def _device_compute_units(self, device: Any) -> int:
        value = self._device_attr(device, "max_compute_units")
        if value is None:
            return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _clear_device_details(self) -> None:
        for label in self._detail_labels.values():
            label.set_label("-")

    def _format_value(self, value: Any) -> str:
        if value is None or value == "":
            return "-"
        return str(value)

    def _format_mhz(self, value: Any) -> str | None:
        if value is None:
            return None
        return f"{value} MHz"

    def _format_sequence(self, value: Any) -> str | None:
        if value is None:
            return None
        return " x ".join(str(item) for item in value)

    def _format_vector_width(self, device: Any, prefix: str) -> str | None:
        widths = []
        for dtype in ("char", "short", "int", "long", "float", "double"):
            value = self._device_attr(device, f"{prefix}_{dtype}")
            if value:
                widths.append(f"{dtype}: {value}")
        return ", ".join(widths) if widths else None

    def _format_bytes(self, value: Any) -> str | None:
        if value is None:
            return None
        size = float(value)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                break
            size /= 1024
        if unit == "B":
            return f"{int(size)} {unit}"
        return f"{size:.1f} {unit}"


if __name__ == "__main__":
    
    for plat in cl.get_platforms():
        print("PLATFORM  : ")
        print("  name    : ", plat.name)
        print("  vendor  : ", plat.vendor)
        print("  version :", plat.version)
        for dev in plat.get_devices():
            print("   DEVICE  :", dev.name)
            print("     type  :", cl.device_type.to_string(dev.type))
        print()
    
    class SelectDeviceApp(Gtk.Application):
        def __init__(self):
            super().__init__(
                application_id="com.qmew.OrbitalEngineer.SelectDevice",
                flags=Gio.ApplicationFlags.FLAGS_NONE,
            )

        def do_activate(self):
            dialog = SelectDeviceWindow()
            dialog.set_application(self)
            dialog.connect("close-request", self._on_close_request)
            dialog.present()

        def _on_close_request(self, dialog: SelectDeviceWindow):
            print(dialog.get_selection())
            self.quit()
            return False

    SelectDeviceApp().run([])
