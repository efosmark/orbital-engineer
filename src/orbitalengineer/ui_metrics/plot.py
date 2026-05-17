#!/usr/bin/env python3
import math

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, GLib # type:ignore

import matplotlib
matplotlib.use("GTK4Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg as FigureCanvas

class PlotWindow(Gtk.ApplicationWindow):
    def __init__(
        self,
        app,
        durations: dict[str, list[tuple[int,float]]],
        refresh_interval_ms=100,
        min_ticks_between_redraws=8,
        max_line_points=180,
        auto_refresh=True,
    ):
        super().__init__(application=app, title="Duration plots-dialog")
        self.set_default_size(800, 500)

        self.durations = durations

        #self.model = model
        self.auto_refresh = bool(auto_refresh)
        self.refresh_interval_ms = max(16, int(refresh_interval_ms))
        self.min_ticks_between_redraws = max(1, int(min_ticks_between_redraws))
        self.max_line_points = max(20, int(max_line_points))
        self.title_font_size = 10
        self.label_font_size = 9
        self.tick_font_size = 8
        self.value_font_size = 7
        self.figure_bg = "#0f1117"
        self.axes_bg = "#181c24"
        self.text_color = "#e6edf3"
        self.grid_color = "#2f3642"
        self.spine_color = "#3a4352"
        self.default_colors = [
            "#7cc7ff",
            "#67d7a5",
            "#ffd166",
            "#f78c6c",
            "#c792ea",
            "#ffadad",
            "#9be564",
            "#89ddff",
        ]

        fig = Figure(figsize=(10, 5), dpi=100, facecolor=self.figure_bg)
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[2.0, 1.6],
            width_ratios=[1.0, 1.0, 1.0],
        )
        self.ax_line = fig.add_subplot(gs[0, :])
        self.ax_avg = fig.add_subplot(gs[1, 0])
        self.ax_count = fig.add_subplot(gs[1, 1], sharey=self.ax_avg)
        self.ax_avg_individual = fig.add_subplot(gs[1, 2], sharey=self.ax_avg)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.93, hspace=0.58, wspace=0.10)
        self.canvas = FigureCanvas(fig)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        root.set_hexpand(True)
        root.set_vexpand(True)
        root.append(self.canvas)
        self.set_child(root)

        self.line_artists = []
        self._closed = False
        self._redraw_queued = False
        self._refresh_source_id = 0
        self._last_redraw_tick = -1

        self.prop_to_color = {}
        self._next_color_idx = 0

        self._style_axes()
        self.redraw_plot()
        self.connect("close-request", self.on_close_request)

    def redraw_plot(self):
        self.ax_line.clear()
        self.ax_avg.clear()
        self.ax_count.clear()
        self.ax_avg_individual.clear()
        self._style_axes()

        line_x_min = None
        line_x_max = None

        self.line_artists = []
        avg_names = []
        avg_values = []
        avg_counts = []
        avg_values_individual = []
        line_y_values = []
        fields_to_plot = list(self.durations.keys())
        for name in fields_to_plot:
            series = self._get_series(name)
            if series is None:
                continue

            x_vals, y_vals, counts = series
            color = self._ensure_field_color(name)
            if x_vals:
                if line_x_min is None:
                    line_x_min = x_vals[0]
                    line_x_max = x_vals[-1]
                else:
                    line_x_min = min(line_x_min, x_vals[0])
                    line_x_max = max(line_x_max, x_vals[-1]) # type: ignore

            y_list = [y * 1000.0 for y in y_vals]
            line_y_values.extend(y_list)
            line_artist, = self.ax_line.plot(
                x_vals,
                y_list,
                label=name,
                color=color,
                linewidth=1.5,
            )
            self.line_artists.append(line_artist)
            if y_list:
                avg_names.append(name)
                avg_values.append(sum(y_list) / len(y_list))
                avg_counts.append(sum(counts) / len(counts))
                raw_data = self.durations.get(name, [])
                raw_y_ms = [y * 1000.0 for _, y in raw_data]
                avg_values_individual.append(sum(raw_y_ms) / len(raw_y_ms))

        if line_x_min is not None and line_x_max is not None:
            if line_x_min == line_x_max:
                self.ax_line.set_xlim(line_x_min - 1, line_x_max + 1)
            else:
                self.ax_line.set_xlim(line_x_min, line_x_max)
        self._set_line_axis_ylim(line_y_values)

        if avg_names:
            bar_colors = [self.prop_to_color[name] for name in avg_names]
            y_pos = list(range(len(avg_names)))
            duration_bars = self.ax_avg.barh(y_pos, avg_values, color=bar_colors)
            count_bars = self.ax_count.barh(y_pos, avg_counts, color=bar_colors)
            duration_individual_bars = self.ax_avg_individual.barh(y_pos, avg_values_individual, color=bar_colors)

            self.ax_avg.set_yticks(y_pos, labels=avg_names)
            self.ax_count.tick_params(axis="y", left=False, labelleft=False)
            self.ax_avg_individual.tick_params(axis="y", left=False, labelleft=False)
            self.ax_avg.invert_yaxis()

            self.ax_avg.margins(y=0.05)
            self.ax_count.margins(y=0.05)
            self.ax_avg_individual.margins(y=0.05)
            self._set_bar_axis_xlim_for_labels(self.ax_avg, avg_values)
            self._set_bar_axis_xlim_for_labels(self.ax_count, avg_counts)
            self._set_bar_axis_xlim_for_labels(self.ax_avg_individual, avg_values_individual)

            for bar, value in zip(duration_bars, avg_values):
                self.ax_avg.text(
                    bar.get_width(),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
            for bar, value in zip(count_bars, avg_counts):
                self.ax_count.text(
                    bar.get_width(),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
            for bar, value in zip(duration_individual_bars, avg_values_individual):
                self.ax_avg_individual.text(
                    bar.get_width(),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
        
        self.canvas.draw_idle()

    def _ensure_field_color(self, name):
        color = self.prop_to_color.get(name)
        if color is not None:
            return color
        color = self.default_colors[self._next_color_idx % len(self.default_colors)]
        self._next_color_idx += 1
        self.prop_to_color[name] = color
        return color

    def _style_axes(self):
        self._style_axis(self.ax_line, show_grid=True)
        self._style_axis(self.ax_avg, show_grid=True)
        self._style_axis(self.ax_count, show_grid=True)
        self._style_axis(self.ax_avg_individual, show_grid=True)

        self.ax_line.set_title("Time durations", fontsize=self.title_font_size, color=self.text_color)
        self.ax_line.set_xlabel("Tick", fontsize=self.label_font_size, color=self.text_color)
        self.ax_line.set_ylabel("Duration (ms)", fontsize=self.label_font_size, color=self.text_color)
        self.ax_line.tick_params(axis="both", labelsize=self.tick_font_size, colors=self.text_color)

        self.ax_avg.set_title("Average duration per tick (summed)", fontsize=self.title_font_size, color=self.text_color)
        self.ax_avg.set_xlabel("Average (ms)", fontsize=self.label_font_size, color=self.text_color)
        self.ax_avg.set_ylabel("")
        self.ax_avg.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_avg.tick_params(axis="y", labelsize=self.tick_font_size, colors=self.text_color)

        self.ax_count.set_title("Average count", fontsize=self.title_font_size, color=self.text_color)
        self.ax_count.set_xlabel("Count", fontsize=self.label_font_size, color=self.text_color)
        self.ax_count.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_count.tick_params(axis="y", left=False, labelleft=False, colors=self.text_color)

        self.ax_avg_individual.set_title("Average duration per sample", fontsize=self.title_font_size, color=self.text_color)
        self.ax_avg_individual.set_xlabel("Average (ms)", fontsize=self.label_font_size, color=self.text_color)
        self.ax_avg_individual.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_avg_individual.tick_params(axis="y", left=False, labelleft=False, colors=self.text_color)

    def _style_axis(self, axis, show_grid=False):
        axis.set_facecolor(self.axes_bg)
        for spine in axis.spines.values():
            spine.set_color(self.spine_color)
        if show_grid:
            axis.grid(True, color=self.grid_color, linewidth=0.6, alpha=0.65)

    def _set_bar_axis_xlim_for_labels(self, axis, values):
        if not values:
            return
        max_value = max(values)
        if max_value <= 0:
            axis.set_xlim(right=1.0)
            return
        axis.set_xlim(right=max_value * 1.16)

    def _set_line_axis_ylim(self, values):
        values = sorted(v for v in values if math.isfinite(v))
        if not values:
            return

        max_value = values[-1]
        if max_value <= 0:
            self.ax_line.set_ylim(0.0, 1.0)
            return

        if len(values) < 8:
            upper = max_value
        else:
            median = self._quantile(values, 0.5)
            p95 = self._quantile(values, 0.95)
            deviations = sorted(abs(v - median) for v in values)
            mad = self._quantile(deviations, 0.5)
            robust_upper = max(p95 * 1.5, median + (8.0 * mad))
            upper = min(max_value, robust_upper) if robust_upper > 0 else max_value

        self.ax_line.set_ylim(0.0, max(upper * 1.12, 0.001))

    def _quantile(self, sorted_values, q):
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = math.floor((len(sorted_values) - 1) * q)
        return sorted_values[idx]

    def _redraw_on_main_thread(self):
        self._redraw_queued = False
        if self._closed:
            return False
        self.redraw_plot()
        return False

    def queue_redraw(self):
        if self._closed:
            return
        if self._redraw_queued:
            return
        self._redraw_queued = True
        GLib.idle_add(self._redraw_on_main_thread)

    def on_close_request(self, _window):
        self._closed = True
        if self._refresh_source_id:
            GLib.source_remove(self._refresh_source_id)
            self._refresh_source_id = 0
        return False

    def _get_series(self, name):
        data = self.durations.get(name)
        if not data:
            return None
        summed_by_x = {}
        counts_by_x = {}
        for x, y in data:
            summed_by_x[x] = summed_by_x.get(x, 0.0) + y
            counts_by_x[x] = counts_by_x.get(x, 0) + 1

        x_vals = list(summed_by_x.keys())
        x_vals.sort()
        y_vals = [summed_by_x[x] for x in x_vals]
        count_vals = [counts_by_x[x] for x in x_vals]

        if len(x_vals) > self.max_line_points:
            step = math.ceil(len(x_vals) / self.max_line_points)
            x_vals = x_vals[::step]
            y_vals = y_vals[::step]
            count_vals = count_vals[::step]
        return x_vals, y_vals, count_vals
