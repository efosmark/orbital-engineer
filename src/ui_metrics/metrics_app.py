import os
import socket
import json

from collections import defaultdict, deque
from typing import Any

from orbitalengineer.ui.gtk4 import Gtk, Gio, GObject, GLib
from ui_metrics import ui_metrics_config
from ui_metrics.plot import PlotWindow
from orbitalengineer.engine.config import METRIC_SOCKET_PATH


class MetricsApp(Gtk.Application):
    props:Any
    durations = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__(application_id=ui_metrics_config.APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.durations = defaultdict(lambda:deque(maxlen=200))
        self.plot_win = None
        self._refresh_source_id = 0
        self._max_metrics_per_poll = ui_metrics_config.MAX_METRICS_PER_POLL
        try:
            os.unlink(METRIC_SOCKET_PATH)
        except FileNotFoundError: pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(METRIC_SOCKET_PATH)
        self.sock.setblocking(False)
        print(f"listening on {METRIC_SOCKET_PATH}")
        self._refresh_source_id = GLib.timeout_add(ui_metrics_config.REFRESH_INTERVAL_MS, self._periodic_refresh)

    def _periodic_refresh(self):
        received_metric = False

        for _ in range(self._max_metrics_per_poll):
            try:
                data, _addr = self.sock.recvfrom(65536)
            except BlockingIOError:
                break

            try:
                metric = json.loads(data)
            except json.JSONDecodeError:
                print("bad metric:", data)
                continue

            tick_id = metric.get("value")
            if tick_id is None:
                continue

            received_metric = True
            for m in metric.get("timeline", []):
                self.add_duration(m["name"], m["duration_ms"], tick_id)
        if received_metric and self.plot_win is not None:
            self.plot_win.queue_redraw()
        return True

    def do_startup(self):
        Gtk.Application.do_startup(self)

    def do_activate(self):
        Gtk.Application.do_activate(self)
        if self.plot_win is not None:
            self.plot_win.present()
            return 

        self.plot_win = PlotWindow(self, self.durations)
        self.plot_win.connect("destroy", self._on_plot_window_destroyed)
        self.plot_win.present()
        self.plot_win.queue_redraw()

    def do_shutdown(self):
        if self._refresh_source_id:
            GLib.source_remove(self._refresh_source_id)
            self._refresh_source_id = 0
        self.sock.close()
        try:
            os.unlink(METRIC_SOCKET_PATH)
        except FileNotFoundError:
            pass
        Gtk.Application.do_shutdown(self)

    def _on_plot_window_destroyed(self, _window):
        self.plot_win = None

    def add_duration(self, name, duration, tick_id):
        if tick_id == 0:
            self.durations.clear()
        self.durations[name].append((tick_id, duration))    

if __name__ == "__main__":
    app = MetricsApp()
    app.run(None)
