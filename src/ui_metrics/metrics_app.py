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
        self.durations = defaultdict(deque)
        self._history_seconds = {
            "gpu_utilization": ui_metrics_config.UTILIZATION_HISTORY_SECONDS,
            "cpu_utilization": ui_metrics_config.UTILIZATION_HISTORY_SECONDS,
            "gpu_temperature": ui_metrics_config.GPU_TEMPERATURE_HISTORY_SECONDS,
            "energy_error": ui_metrics_config.ENERGY_HISTORY_SECONDS,
            "total_KE": ui_metrics_config.ENERGY_HISTORY_SECONDS,
            "total_GPE": ui_metrics_config.ENERGY_HISTORY_SECONDS,
            "dt_step": ui_metrics_config.DT_STEP_HISTORY_SECONDS,
            "num_steps": ui_metrics_config.NUM_STEPS_HISTORY_SECONDS,
            "virial_ratio": ui_metrics_config.VIRIAL_RATIO_HISTORY_SECONDS,
        }
        # Samples contain tick ID, simulation time, and value.
        self.metric_series = {}
        self._last_tick_id = None
        self._last_sim_time = None
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
            sim_time = metric.get("sim_time")
            if (
                tick_id == 0
                or (self._last_tick_id is not None and tick_id < self._last_tick_id)
                or (
                    sim_time is not None
                    and self._last_sim_time is not None
                    and sim_time < self._last_sim_time
                )
            ):
                self.clear_metrics()
            self._last_tick_id = tick_id
            if sim_time is not None:
                self._last_sim_time = sim_time
            for name in self._history_seconds:
                value = metric.get(name)
                if value is not None and sim_time is not None:
                    if name not in self.metric_series:
                        self.metric_series[name] = deque()
                    self.metric_series[name].append((tick_id, sim_time, value))
            if sim_time is not None:
                for m in metric.get("timeline", []):
                    self.add_duration(m["name"], m["duration_ms"], tick_id, sim_time)
                self._expire_history(sim_time)
        if received_metric and self.plot_win is not None:
            self.plot_win.queue_redraw()
        return True

    def clear_metrics(self, *, force_redraw=False):
        self.durations.clear()
        self.metric_series.clear()
        self._last_tick_id = None
        self._last_sim_time = None
        if self.plot_win is not None:
            self.plot_win.reset_plot(force_redraw=force_redraw)

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

    def add_duration(self, name, duration, tick_id, sim_time):
        self.durations[name].append((tick_id, sim_time, duration))

    def _expire_history(self, sim_time):
        for history in (self.durations, self.metric_series):
            for name in list(history):
                limit = (
                    ui_metrics_config.DURATION_HISTORY_SECONDS
                    if history is self.durations
                    else self._history_seconds[name]
                )
                oldest_time = sim_time - limit
                samples = history[name]
                while samples and samples[0][1] < oldest_time:
                    samples.popleft()
                if not samples:
                    del history[name]

if __name__ == "__main__":
    app = MetricsApp()
    app.run(None)
