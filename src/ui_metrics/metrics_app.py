import os
import socket
import json

from collections import defaultdict, deque
from typing import Any

from gui.gtk4 import Gtk, Gio, GObject, GLib
from ui_metrics import ui_metrics_config
from ui_metrics.plot import PlotWindow
from orbitalengineer.engine.config import METRIC_SOCKET_PATH


class MetricsApp(Gtk.Application):
    props:Any
    durations = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__(application_id=ui_metrics_config.APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.durations = defaultdict(lambda:deque(maxlen=ui_metrics_config.MAX_DURATION_SAMPLES))
        self._metric_limits = {
            "gpu_utilization": ui_metrics_config.MAX_UTILIZATION_SAMPLES,
            "cpu_utilization": ui_metrics_config.MAX_UTILIZATION_SAMPLES,
            "energy_error": ui_metrics_config.MAX_ENERGY_SAMPLES,
            "total_KE": ui_metrics_config.MAX_ENERGY_SAMPLES,
            "total_GPE": ui_metrics_config.MAX_ENERGY_SAMPLES,
            "dt_step": ui_metrics_config.MAX_DT_STEP_SAMPLES,
        }
        self.metric_series = {}
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
                print("bad metric")
                continue

            if metric.get("name", "tick") != "tick":
                continue

            tick_id = metric.get("value")
            if tick_id is None:
                continue

            received_metric = True
            if tick_id == 0:
                self.durations.clear()
                self.metric_series.clear()
            for name in self._metric_limits:
                value = metric.get(name)
                if value is not None:
                    if name not in self.metric_series:
                        self.metric_series[name] = deque(maxlen=self._metric_limits[name])
                    self.metric_series[name].append((tick_id, value))
            for m in metric.get("timeline", []):
                self.add_duration(m["name"], m["duration_ms"], tick_id)
            self._expire_history(tick_id)
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

        self.plot_win = PlotWindow(self, self.durations, metric_series=self.metric_series)
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
        self.durations[name].append((tick_id, duration))    

    def _expire_history(self, tick_id):
        for history in (self.durations, self.metric_series):
            for name in list(history):
                limit = (
                    ui_metrics_config.MAX_DURATION_SAMPLES
                    if history is self.durations
                    else self._metric_limits[name]
                )
                oldest_tick = tick_id - limit + 1
                samples = history[name]
                while samples and samples[0][0] < oldest_tick:
                    samples.popleft()
                if not samples:
                    del history[name]

if __name__ == "__main__":
    app = MetricsApp()
    app.run(None)
