#!/usr/bin/env python3
from collections import deque
import math

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, GLib # type:ignore

import matplotlib
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg as FigureCanvas
matplotlib.use("GTK4Agg")

from ui_metrics import ui_metrics_config

class PlotWindow(Gtk.ApplicationWindow):
    def __init__(
        self,
        app,
        durations: dict[str, deque[tuple[int, float, float]]],
        metric_series=None,
    ):
        super().__init__(application=app, title=ui_metrics_config.WINDOW_TITLE)
        self.set_default_size(*ui_metrics_config.WINDOW_DEFAULT_SIZE)
        self.durations = durations
        self.metric_series = metric_series if metric_series is not None else {}
        self._paused = False

        header = Gtk.HeaderBar()
        header.set_title_widget(Gtk.Label(label=ui_metrics_config.WINDOW_TITLE))
        self.clear_button = Gtk.Button(label="Clear")
        self.clear_button.set_tooltip_text("Clear all recorded metrics")
        self.clear_button.connect("clicked", lambda _button: app.clear_metrics(force_redraw=True))
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.append(self.clear_button)
        self.pause_button = Gtk.ToggleButton(label="Pause")
        self.pause_button.set_tooltip_text("Pause graph updates while continuing to receive metrics")
        self.pause_button.connect("toggled", self._on_pause_toggled)
        controls.append(self.pause_button)
        header.pack_start(controls)
        self.set_titlebar(header)

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

        window_width, window_height = ui_metrics_config.WINDOW_DEFAULT_SIZE
        fig = Figure(figsize=(window_width / 100, window_height / 100), dpi=100, facecolor=self.figure_bg)
        metrics_gs = fig.add_gridspec(
            1, 3, left=0.08, right=0.98, bottom=0.77, top=0.95, wspace=0.20,
        )
        self.ax_utilization = fig.add_subplot(metrics_gs[0, 0])
        self.ax_gpu_temperature = fig.add_subplot(metrics_gs[0, 1])
        self.ax_dt_step = fig.add_subplot(metrics_gs[0, 2])
        energy_gs = fig.add_gridspec(
            1, 3, left=0.08, right=0.98, bottom=0.49, top=0.67, wspace=0.20,
        )
        self.ax_energy_error = fig.add_subplot(energy_gs[0, 0])
        self.ax_total_ke = fig.add_subplot(energy_gs[0, 1])
        self.ax_total_gpe = self.ax_total_ke.twinx()
        self.ax_virial_ratio = fig.add_subplot(energy_gs[0, 2])
        gs = fig.add_gridspec(
            1, 3,
            left=0.26, right=0.98, bottom=0.08, top=0.39, wspace=0.08,
            width_ratios=[1.0, 1.0, 0.4],
        )
        self.ax_avg_individual = fig.add_subplot(gs[0, 0])
        self.ax_avg = fig.add_subplot(gs[0, 1], sharey=self.ax_avg_individual)
        self.ax_count = fig.add_subplot(gs[0, 2], sharey=self.ax_avg_individual)
        self._row_grids = (metrics_gs, energy_gs, gs)
        self._graph_groups = [
            ("All graphs", tuple(fig.axes)),
            ("Utilization", (self.ax_utilization,)),
            ("GPU temperature", (self.ax_gpu_temperature,)),
            ("Step duration", (self.ax_dt_step,)),
            ("Energy error", (self.ax_energy_error,)),
            ("Total energy", (self.ax_total_ke, self.ax_total_gpe)),
            ("Virial ratio", (self.ax_virial_ratio,)),
            ("Average duration per sample", (self.ax_avg_individual,)),
            ("Average duration per tick", (self.ax_avg,)),
            ("Average count", (self.ax_count,)),
        ]
        self._default_positions = {axis: axis.get_position().frozen() for axis in fig.axes}
        self.graph_selector = Gtk.DropDown.new_from_strings([name for name, _axes in self._graph_groups])
        self.graph_selector.set_tooltip_text("Choose a graph to expand, or show all graphs")
        self.graph_selector.connect("notify::selected", self._on_graph_selected)
        header.pack_end(self.graph_selector)
        self.canvas = FigureCanvas(fig)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
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

    def _on_graph_selected(self, _selector, _param):
        self._apply_graph_layout()
        self.canvas.draw_idle()

    def _on_canvas_resize(self, _event):
        self._apply_graph_layout()
        self.canvas.draw_idle()

    def _layout_margins(self):
        fig = self.canvas.figure
        width, height = fig.bbox.width, fig.bbox.height
        unit = fig.dpi / 72
        font = FontProperties(family="monospace", size=self.tick_font_size)
        renderer = self.canvas.get_renderer()
        label_width = max(
            (renderer.get_text_width_height_descent(name, font, False)[0] for name in self.durations),
            default=0,
        )
        left = min(36 * unit / width, 0.20)
        bar_left = min(max(16 * unit + label_width, 36 * unit) / width, 0.45)
        right = 1 - min(16 * unit / width, 0.10)
        bottom = min(38 * unit / height, 0.12)
        top = 1 - min(28 * unit / height, 0.10)
        gap = min(58 * unit / height, 0.14)
        row_height = (top - bottom - 2 * gap) / 3.7
        row_top = top
        for grid, weight, row_left, gap_points in zip(
            self._row_grids, (1, 1, 1.7), (left, left, bar_left), (84, 84, 14),
        ):
            row_bottom = row_top - weight * row_height
            # Keep label space in physical units instead of wasting a percentage of wide windows.
            gap_width = min(gap_points * unit / width, (right - row_left) / 4)
            avg_width = (right - row_left - 2 * gap_width) / 3
            grid.update(left=row_left, right=right, bottom=row_bottom, top=row_top, wspace=gap_width / avg_width)
            row_top = row_bottom - gap
        self._default_positions = {
            axis: axis.get_subplotspec().get_position(fig).frozen()
            for axis in fig.axes
        }
        return left, bar_left, right, bottom, top

    def _apply_graph_layout(self):
        left, bar_left, right, bottom, top = self._layout_margins()
        selected = self.graph_selector.get_selected()
        _name, axes = self._graph_groups[selected]
        for axis, position in self._default_positions.items():
            axis.set_visible(selected == 0 or axis in axes)
            axis.set_position(position)
        for axis in (self.ax_avg, self.ax_count):
            axis.tick_params(axis="y", left=False, labelleft=False)
        if selected:
            duration_axes = (self.ax_avg_individual, self.ax_avg, self.ax_count)
            for axis in axes:
                if axis in duration_axes:
                    axis.set_position((bar_left, bottom, right - bar_left, top - bottom))
                    axis.tick_params(axis="y", left=True, labelleft=True)
                else:
                    plot_right = right
                    if self.ax_total_gpe in axes:
                        plot_right = 1 - left
                    axis.set_position((left, bottom, plot_right - left, top - bottom))

    def _on_pause_toggled(self, button):
        self._paused = button.get_active()
        button.set_label("Resume" if self._paused else "Pause")
        if not self._paused:
            self.queue_redraw()

    def reset_plot(self, *, force_redraw=False):
        self.prop_to_color.clear()
        self._next_color_idx = 0
        self._last_redraw_tick = -1
        if force_redraw or not self._paused:
            self.redraw_plot()

    def redraw_plot(self):
        self.ax_avg.clear()
        self.ax_count.clear()
        self.ax_avg_individual.clear()
        self.ax_utilization.clear()
        self.ax_gpu_temperature.clear()
        self.ax_energy_error.clear()
        self.ax_dt_step.clear()
        self.ax_virial_ratio.clear()
        self.ax_total_ke.clear()
        self.ax_total_gpe.clear()
        self._style_axes()
        self._update_energy_error_scale()

        for name, axis, color in (
            ("gpu_utilization", self.ax_utilization, self.default_colors[0]),
            ("cpu_utilization", self.ax_utilization, self.default_colors[3]),
            ("gpu_temperature", self.ax_gpu_temperature, self.default_colors[3]),
            ("energy_error", self.ax_energy_error, self.default_colors[1]),
            ("dt_step", self.ax_dt_step, self.default_colors[2]),
            ("virial_ratio", self.ax_virial_ratio, self.default_colors[4]),
            ("total_KE", self.ax_total_ke, self.default_colors[0]),
            ("total_GPE", self.ax_total_gpe, self.default_colors[4]),
        ):
            data = self.metric_series.get(name)
            if data:
                _tick_ids, x_vals, y_vals = zip(*data)
                if name == "dt_step":
                    y_vals = [value * 1000 for value in y_vals]
                label = {
                    "gpu_utilization": "GPU", "cpu_utilization": "CPU",
                    "total_KE": "KE", "total_GPE": "GPE",
                }.get(name)
                axis.plot(x_vals, y_vals, color=color, linewidth=1.2, label=label)

        if self.ax_utilization.lines:
            self.ax_utilization.legend(
                loc="upper right", fontsize=self.tick_font_size,
                facecolor=self.axes_bg, edgecolor=self.spine_color,
                labelcolor=self.text_color,
            )

        energy_lines = list(self.ax_total_ke.lines) + list(self.ax_total_gpe.lines)
        if energy_lines:
            self.ax_total_ke.legend(
                energy_lines, [line.get_label() for line in energy_lines],
                loc="upper left", fontsize=self.tick_font_size,
                facecolor=self.axes_bg, edgecolor=self.spine_color,
                labelcolor=self.text_color,
            )

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

            y_list = [y for y in y_vals]
            line_y_values.extend(y_list)
            if y_list:
                avg_names.append(name)
                avg_values.append(sum(y_list) / len(y_list))
                avg_counts.append(sum(counts) / len(counts))
                raw_data = self.durations.get(name, [])
                raw_y_ms = [y for _, _, y in raw_data]
                avg_values_individual.append(sum(raw_y_ms) / len(raw_y_ms))

        if avg_names:
            rows = sorted(
                zip(avg_names, avg_values, avg_counts, avg_values_individual),
                key=lambda row: row[3],
                reverse=True,
            )
            avg_names, avg_values, avg_counts, avg_values_individual = map(list, zip(*rows))
            bar_colors = [self.prop_to_color[name] for name in avg_names]
            y_pos = list(range(len(avg_names)))
            duration_bars = self.ax_avg.barh(y_pos, avg_values, color=bar_colors)
            count_bars = self.ax_count.barh(y_pos, avg_counts, color=bar_colors)
            duration_individual_bars = self.ax_avg_individual.barh(y_pos, avg_values_individual, color=bar_colors)

            self.ax_avg_individual.set_yticks(y_pos, labels=avg_names)
            self.ax_count.tick_params(axis="y", left=False, labelleft=False)
            self.ax_avg.tick_params(axis="y", left=False, labelleft=False)
            self.ax_avg_individual.invert_yaxis()

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
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
            for bar, value in zip(count_bars, avg_counts):
                self.ax_count.text(
                    bar.get_width(),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
            for bar, value in zip(duration_individual_bars, avg_values_individual):
                self.ax_avg_individual.text(
                    bar.get_width(),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=self.value_font_size,
                    color=self.text_color,
                )
        
        self._apply_graph_layout()
        self.canvas.draw_idle()

    def _ensure_field_color(self, name):
        color = self.prop_to_color.get(name)
        if color is not None:
            return color
        color = self.default_colors[self._next_color_idx % len(self.default_colors)]
        self._next_color_idx += 1
        self.prop_to_color[name] = color
        return color

    def _update_energy_error_scale(self):
        samples = self.metric_series.get("energy_error", [])
        magnitudes = sorted(abs(value) for _, _, value in samples if math.isfinite(value))
        peak = 0.0
        if len(magnitudes) >= ui_metrics_config.ENERGY_ERROR_SCALE_MIN_SAMPLES:
            # Use an observed percentile without interpolating toward an isolated spike.
            index = math.floor((len(magnitudes) - 1) * ui_metrics_config.ENERGY_ERROR_SCALE_QUANTILE)
            peak = magnitudes[index]
        limit = peak if peak > 0 else ui_metrics_config.ENERGY_ERROR_ZERO_RANGE
        self.ax_energy_error.set_ylim(-limit, limit)
        self.ax_energy_error.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)

    def _style_axes(self):
        for axis, title in (
            (self.ax_utilization, "Utilization (%)"),
            (self.ax_gpu_temperature, "GPU temperature (°C)"),
            (self.ax_energy_error, "Energy error (ΔE / E₀)"),
            (self.ax_dt_step, "dt_step (ms)"),
            (self.ax_virial_ratio, "Virial ratio"),
            (self.ax_total_ke, "Total energy"),
        ):
            self._style_axis(axis, show_grid=True)
            axis.margins(x=0)
            axis.set_title(title, fontsize=self.title_font_size, color=self.text_color)
            axis.set_xlabel("Sim Time", fontsize=self.label_font_size, color=self.text_color)
            axis.set_ylabel("")
            axis.tick_params(axis="both", labelsize=self.tick_font_size, colors=self.text_color)

        self.ax_utilization.set_ylim(0, 100)
        self.ax_gpu_temperature.set_ylim(25, 100)

        self._style_axis(self.ax_total_gpe)
        self.ax_total_gpe.margins(x=0)
        self.ax_total_gpe.yaxis.set_label_position("right")
        self.ax_total_gpe.yaxis.tick_right()
        self.ax_total_gpe.yaxis.set_offset_position("right")
        self.ax_total_gpe.set_ylabel("")
        self.ax_total_gpe.tick_params(
            axis="y", left=False, labelleft=False, right=True, labelright=True,
            labelsize=self.tick_font_size, colors=self.default_colors[4],
        )
        self.ax_total_gpe.yaxis.get_offset_text().set_color(self.default_colors[4])
        self.ax_total_ke.yaxis.label.set_color(self.default_colors[0])
        self.ax_total_ke.tick_params(axis="y", colors=self.default_colors[0])
        self.ax_total_ke.yaxis.get_offset_text().set_color(self.default_colors[0])

        self._style_axis(self.ax_avg, show_grid=True)
        self._style_axis(self.ax_count, show_grid=True)
        self._style_axis(self.ax_avg_individual, show_grid=True)

        self.ax_avg.set_title("Average duration per tick (summed)", fontsize=self.title_font_size, color=self.text_color)
        self.ax_avg.set_xlabel("Average (ms)", fontsize=self.label_font_size, color=self.text_color)
        self.ax_avg.set_ylabel("")
        self.ax_avg.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_avg.tick_params(axis="y", left=False, labelleft=False, colors=self.text_color)

        self.ax_count.set_title("Average count", fontsize=self.title_font_size, color=self.text_color)
        self.ax_count.set_xlabel("Count", fontsize=self.label_font_size, color=self.text_color)
        self.ax_count.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_count.tick_params(axis="y", left=False, labelleft=False, colors=self.text_color)

        self.ax_avg_individual.set_title("Average duration per sample", fontsize=self.title_font_size, color=self.text_color)
        self.ax_avg_individual.set_xlabel("Average (ms)", fontsize=self.label_font_size, color=self.text_color)
        self.ax_avg_individual.tick_params(axis="x", labelsize=self.tick_font_size, colors=self.text_color)
        self.ax_avg_individual.tick_params(axis="y", labelsize=self.tick_font_size, colors=self.text_color, labelfontfamily="monospace")

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

    def _quantile(self, sorted_values, q):
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = math.floor((len(sorted_values) - 1) * q)
        return sorted_values[idx]

    def _redraw_on_main_thread(self):
        self._redraw_queued = False
        if self._closed or self._paused:
            return False
        self.redraw_plot()
        return False

    def queue_redraw(self):
        if self._closed or self._paused:
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
        for x, _sim_time, y in data:
            summed_by_x[x] = summed_by_x.get(x, 0.0) + y
            counts_by_x[x] = counts_by_x.get(x, 0) + 1

        x_vals = list(summed_by_x.keys())
        x_vals.sort()
        y_vals = [summed_by_x[x] for x in x_vals]
        count_vals = [counts_by_x[x] for x in x_vals]

        return x_vals, y_vals, count_vals
