#!/usr/bin/env python3
from collections import deque
import math

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, GLib # type:ignore

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg as FigureCanvas
matplotlib.use("GTK4Agg")

from ui_metrics import ui_metrics_config

class PlotWindow(Gtk.ApplicationWindow):
    def __init__(
        self,
        app,
        durations: dict[str, deque[tuple[int, float, float]]],
        metric_series: dict[str, deque[tuple[int, float, float]]] | None = None,
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
        self.figure_bg = "#0f1117"
        background_css = Gtk.CssProvider()
        background_css.load_from_data(f"window {{ background-color: {self.figure_bg}; }}".encode())
        self.get_style_context().add_provider(background_css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
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
        fig = Figure(
            figsize=(window_width / 100, window_height / 100), dpi=100,
            facecolor=self.figure_bg,
        )
        # Give each row 12px above and below its labels, leaving a visible 24px gap.
        self._layout_engine = ConstrainedLayoutEngine(w_pad=8 / fig.dpi, h_pad=12 / fig.dpi, hspace=0)
        fig.set_layout_engine(self._layout_engine)
        rows = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.6])
        metrics_gs = rows[0].subgridspec(1, 4)
        self.ax_utilization = fig.add_subplot(metrics_gs[0, 0])
        self.ax_gpu_temperature = fig.add_subplot(metrics_gs[0, 1])
        self.ax_dt_step = fig.add_subplot(metrics_gs[0, 2])
        self.ax_num_steps = fig.add_subplot(metrics_gs[0, 3])
        energy_gs = rows[1].subgridspec(1, 3)
        self.ax_energy_error = fig.add_subplot(energy_gs[0, 0])
        self.ax_total_ke = fig.add_subplot(energy_gs[0, 1])
        self.ax_total_gpe = self.ax_total_ke.twinx()
        self.ax_virial_ratio = fig.add_subplot(energy_gs[0, 2])
        self.ax_avg_individual = fig.add_subplot(rows[2])
        self._graph_groups = [
            ("Utilization", (self.ax_utilization,)),
            ("GPU temperature", (self.ax_gpu_temperature,)),
            ("Step duration", (self.ax_dt_step,)),
            ("Steps per tick", (self.ax_num_steps,)),
            ("Energy error", (self.ax_energy_error,)),
            ("Total energy", (self.ax_total_ke, self.ax_total_gpe)),
            ("Virial ratio", (self.ax_virial_ratio,)),
            ("Average duration per sample", (self.ax_avg_individual,)),
        ]
        self._layout_selection = None
        self._empty_graphs_message = fig.text(
            0.5, 0.5, "Select graphs from the Graphs menu", ha="center", va="center",
            color=self.text_color, fontsize=self.title_font_size, visible=False,
        )
        self.graph_selector = Gtk.MenuButton(label="Graphs")
        self.graph_selector.set_tooltip_text("Choose which graphs to show")
        popover = Gtk.Popover()
        graph_options = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        graph_options.set_margin_start(12)
        graph_options.set_margin_end(12)
        graph_options.set_margin_top(12)
        graph_options.set_margin_bottom(12)
        self._graph_toggles = []
        for name, _axes in self._graph_groups:
            toggle = Gtk.CheckButton(label=name)
            toggle.set_active(True)
            toggle.connect("toggled", self._on_graph_toggled)
            graph_options.append(toggle)
            self._graph_toggles.append(toggle)
        popover.set_child(graph_options)
        self.graph_selector.set_popover(popover)
        header.pack_end(self.graph_selector)
        self.canvas = FigureCanvas(fig)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        self._duration_layout_names: tuple[str, ...] | None = None
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        root.set_margin_start(12)
        root.set_margin_end(12)
        root.set_margin_top(12)
        root.set_margin_bottom(12)
        root.set_hexpand(True)
        root.set_vexpand(True)
        root.append(self.canvas)
        self.set_child(root)

        self._closed = False
        self._redraw_queued = False
        self._refresh_source_id = 0

        self.prop_to_color = {}
        self._next_color_idx = 0

        self._style_axes()
        self.redraw_plot()
        self.connect("close-request", self.on_close_request)

    def _on_graph_toggled(self, _toggle):
        self._apply_graph_layout()
        self.canvas.draw_idle()

    def _request_layout(self):
        self.canvas.figure.set_layout_engine(self._layout_engine)

    def _on_canvas_draw(self, _event):
        # Keep automatic layout's result until something structural changes.
        # Scrolling tick labels must not resize the plotting areas on each frame.
        if self.canvas.figure.get_layout_engine() is self._layout_engine:
            self.canvas.figure.set_layout_engine("none")

    def _on_canvas_resize(self, _event):
        self._request_layout()
        self.canvas.draw_idle()

    def _ensure_y_tick_space(self):
        if self.canvas.figure.get_layout_engine() is self._layout_engine:
            return
        renderer = self.canvas.get_renderer()
        previous = None
        for _name, axes in self._graph_groups:
            if not axes[0].get_visible():
                continue
            position = axes[0].get_position()
            left = axes[0].get_window_extent(renderer).x0
            right = axes[0].get_window_extent(renderer).x1
            for axis in axes:
                bounds = axis.yaxis.get_tightbbox(renderer)
                if bounds is not None:
                    left = min(left, bounds.x0)
                    right = max(right, bounds.x1)
            # Reflow only when y-axis text outgrows its space. Ignore scrolling x labels.
            if (
                left < 4
                or right > self.canvas.figure.bbox.width - 4
                or (previous is not None and abs(previous[0] - position.y0) < 1e-6 and left < previous[1] + 4)
            ):
                self._request_layout()
                return
            previous = position.y0, right

    def _apply_graph_layout(self):
        selection = tuple(toggle.get_active() for toggle in self._graph_toggles)
        for visible, (_name, axes) in zip(selection, self._graph_groups):
            for axis in axes:
                axis.set_visible(visible)
                axis.set_in_layout(visible)
        self._empty_graphs_message.set_visible(not any(selection))
        if selection == self._layout_selection:
            return
        self._layout_selection = selection
        self._request_layout()
        self.graph_selector.set_label(f"Graphs ({sum(selection)}/{len(selection)})")
        visible_rows = []
        for indices, weight in ((range(4), 1), (range(4, 7), 1), (range(7, 8), 1.6)):
            groups = [self._graph_groups[index] for index in indices if selection[index]]
            if groups:
                visible_rows.append((groups, weight))
        if not visible_rows:
            return
        rows = self.canvas.figure.add_gridspec(
            len(visible_rows), 1, height_ratios=[weight for _groups, weight in visible_rows],
        )
        for row, (groups, _weight) in enumerate(visible_rows):
            columns = rows[row].subgridspec(1, len(groups))
            for column, (_name, axes) in enumerate(groups):
                for axis in axes:
                    axis.set_subplotspec(columns[0, column])

    def _on_pause_toggled(self, button):
        self._paused = button.get_active()
        button.set_label("Resume" if self._paused else "Pause")
        if not self._paused:
            self.queue_redraw()

    def reset_plot(self, *, force_redraw=False):
        self.prop_to_color.clear()
        self._next_color_idx = 0
        self._request_layout()
        if force_redraw or not self._paused:
            self.redraw_plot()

    def redraw_plot(self):
        self.ax_avg_individual.clear()
        self.ax_utilization.clear()
        self.ax_gpu_temperature.clear()
        self.ax_energy_error.clear()
        self.ax_dt_step.clear()
        self.ax_num_steps.clear()
        self.ax_virial_ratio.clear()
        self.ax_total_ke.clear()
        self.ax_total_gpe.clear()
        self._style_axes()

        for name, axis, color in (
            ("gpu_utilization", self.ax_utilization, self.default_colors[0]),
            ("cpu_utilization", self.ax_utilization, self.default_colors[3]),
            ("gpu_temperature", self.ax_gpu_temperature, self.default_colors[3]),
            ("energy_error", self.ax_energy_error, self.default_colors[1]),
            ("dt_step", self.ax_dt_step, self.default_colors[2]),
            ("num_steps", self.ax_num_steps, self.default_colors[5]),
            ("virial_ratio", self.ax_virial_ratio, self.default_colors[4]),
            ("total_KE", self.ax_total_ke, self.default_colors[0]),
            ("total_GPE", self.ax_total_gpe, self.default_colors[4]),
        ):
            data = self.metric_series.get(name)
            if data:
                _tick_ids, x_vals, y_vals = zip(*data)
                if name == "energy_error":
                    self._set_energy_error_range(y_vals)
                if name == "dt_step":
                    y_vals = [value * 1000 for value in y_vals]
                label = {
                    "gpu_utilization": "GPU", "cpu_utilization": "CPU",
                    "total_KE": "KE", "total_GPE": "GPE",
                }.get(name)
                axis.plot(x_vals, y_vals, color=color, linewidth=1.2, label=label)

        steps = self.metric_series.get("num_steps", [])
        max_steps = max((value for _, _, value in steps if math.isfinite(value)), default=0)
        self.ax_num_steps.set_ylim(0, max(5, math.ceil(max_steps) + 1))

        for axis, reference in ((self.ax_energy_error, 0), (self.ax_virial_ratio, 1)):
            # Artists do not contribute to data limits; off-screen references are clipped.
            axis.add_artist(Line2D(
                [0, 1], [reference, reference], transform=axis.get_yaxis_transform(),
                color=self.text_color, linestyle=":", linewidth=1, alpha=0.7,
            ))

        if self.ax_utilization.lines:
            self.ax_utilization.legend(
                loc="upper right", fontsize=self.tick_font_size,
                facecolor=self.axes_bg, edgecolor=self.spine_color,
                labelcolor=self.text_color,
            )

        energy_lines = list(self.ax_total_ke.lines) + list(self.ax_total_gpe.lines)
        if energy_lines:
            self.ax_total_ke.legend(
                energy_lines, [line.get_label() for line in energy_lines], #type:ignore
                loc="upper left", fontsize=self.tick_font_size,
                facecolor=self.axes_bg, edgecolor=self.spine_color,
                labelcolor=self.text_color,
            )

        self._draw_duration_bars()

        self._apply_graph_layout()
        self._ensure_y_tick_space()
        self.canvas.draw_idle()

    def _set_energy_error_range(self, values):
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return
        low, high = min(finite_values), max(finite_values)
        peak = max(abs(low), abs(high))
        if peak == 0:
            self.ax_energy_error.set_ylim(-1, 1)
            return
        magnitude = 10 ** math.floor(math.log10(peak))
        lower = math.floor(low / magnitude) * magnitude
        upper = math.ceil(high / magnitude) * magnitude
        if lower == upper:
            lower -= magnitude
            upper += magnitude
        self.ax_energy_error.set_ylim(lower, upper)

    def _ensure_field_color(self, name):
        color = self.prop_to_color.get(name)
        if color is not None:
            return color
        color = self.default_colors[self._next_color_idx % len(self.default_colors)]
        self._next_color_idx += 1
        self.prop_to_color[name] = color
        return color

    def _draw_duration_bars(self):
        rows: list[tuple[str, float, float]] = []
        for name, samples in self.durations.items():
            if not samples:
                continue
            average_ms = sum(value for _, _, value in samples) / len(samples)
            if average_ms <= 0:
                continue
            tick_count = len({tick for tick, _, _ in samples})
            average_count = len(samples) / tick_count
            rows.append((name, average_ms, average_count))
        rows.sort(key=lambda row: row[1], reverse=True)
        layout_names = tuple(sorted(name for name, _average, _count in rows))
        if layout_names != self._duration_layout_names:
            self._duration_layout_names = layout_names
            self._request_layout()
        axis = self.ax_avg_individual
        if not rows:
            axis.set_yticks([])
            axis.set_xlim(0, 1)
            axis.text(
                0.5, 0.5, "Waiting for timing samples", transform=axis.transAxes,
                ha="center", va="center", fontsize=self.label_font_size, color=self.text_color,
            )
            return

        names = [name for name, _average, _count in rows]
        averages = [average for _name, average, _count in rows]
        counts = [count for _name, _average, count in rows]
        bars = axis.barh(
            range(len(rows)), averages, height=0.65,
            color=[self._ensure_field_color(name) for name in names],
        )
        axis.set_yticks(range(len(rows)), labels=names)
        axis.invert_yaxis()
        axis.margins(y=0.04)
        axis.set_xlim(0, max(averages) * 1.4 if max(averages) > 0 else 1)
        for bar, average, count in zip(bars, averages, counts):
            axis.annotate(
                f"{average:.3g} ms  ·  avg count {count:.1f}/tick",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                fontsize=self.tick_font_size, color=self.text_color,
            )

    def _style_axes(self):
        for axis, title in (
            (self.ax_utilization, "Utilization (%)"),
            (self.ax_gpu_temperature, "GPU temperature (°C)"),
            (self.ax_energy_error, "Energy error (ΔE / E₀)"),
            (self.ax_dt_step, "dt_step (ms)"),
            (self.ax_num_steps, "Steps per tick"),
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
        self.ax_energy_error.ticklabel_format(
            axis="y", style="sci", scilimits=(-3, 3), useOffset=False, useMathText=False,
        )
        self.ax_gpu_temperature.set_ylim(25, 100)
        self.ax_num_steps.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

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

        self._style_axis(self.ax_avg_individual, show_grid=True)
        self.ax_avg_individual.set_axisbelow(True)
        self.ax_avg_individual.grid(False, axis="y")
        self.ax_avg_individual.spines["top"].set_visible(False)
        self.ax_avg_individual.spines["right"].set_visible(False)

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
