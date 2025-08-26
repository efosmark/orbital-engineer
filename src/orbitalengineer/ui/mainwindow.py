from collections import deque
import os
import statistics
import threading
import time

import numpy as np
os.sched_setaffinity(0, {0, 1, 2})

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalnp import OrbitalSimController
from orbitalengineer.ui.gtk4 import Gtk, Gdk, Gio, GLib #, Adw
from orbitalengineer.ui import canvas


APP_ID = "com.qmew.OrbitalEngineer"
WINDOW_DEFAULT_SIZE = (800, 600)


class App(Gtk.Application):

    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.pressed_keys = set()
        self.paused = False
        self.start_maximized = False

        self.tick = 0
        self._tick_duration = deque(maxlen=100)
        self._tick_steps = deque(maxlen=100)
        
        self.orbital = OrbitalSimController()
        self.canvas = canvas.OrbitalCanvas(self.orbital)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        self.canvas.set_halign(Gtk.Align.FILL)
        self.canvas.set_valign(Gtk.Align.FILL)
        
        self.__init_menu_actions()
    
    def do_startup(self):
        Gtk.Application.do_startup(self)
    
    def start_simulation(self):
        self.orbital.init_sim()
        GLib.idle_add(self.idle_loop)
    
    def __init_menu_actions(self):
        def add_toggle_action(name: str, initial: bool, handler):
            action = Gio.SimpleAction.new_stateful(
                name, None, GLib.Variant.new_boolean(initial)
            )
            action.connect("change-state", handler)
            self.add_action(action)

        # Two stateful (boolean) actions to back the checkable menu items.
        add_toggle_action("pause", False, self.on_toggle_changed)
        add_toggle_action("show_grid", self.canvas.show_map, self.on_toggle_changed)
        add_toggle_action("show_force_vectors", self.canvas.show_force_vectors, self.on_toggle_changed)
        add_toggle_action("show_history", self.canvas.show_history, self.on_toggle_changed)
        add_toggle_action("show_orbital_ellipse", self.canvas.show_orbital_ellipse, self.on_toggle_changed)
        add_toggle_action("track_focused", self.canvas.track_focused, self.on_toggle_changed)
        add_toggle_action("show_magnifier", self.canvas.show_magnifier, self.on_toggle_changed)

        # A regular (stateless) action just to show contrast.
        act_quit = Gio.SimpleAction.new("quit", None)
        act_quit.connect("activate", lambda a, p: self.quit())
        self.add_action(act_quit)
        
        # Optional accelerator for quit
        self.set_accels_for_action("app.quit", ["<Primary>q"])
        self.set_accels_for_action("app.show_grid", ["g"])
        self.set_accels_for_action("app.show_force_vectors", ["v"])
        self.set_accels_for_action("app.show_orbital_ellipse", ["o"])
        self.set_accels_for_action("app.show_history", ["h"])
        self.set_accels_for_action("app.track_focused", ["t"])
        self.set_accels_for_action("app.show_magnifier", ["m"])

    def on_toggle_changed(self, action: Gio.SimpleAction, value: GLib.Variant):
        action.set_state(value)
        name = action.get_name()
        state = value.get_boolean()
        if name == "show_grid":
            self.canvas.show_map = state
        elif name == "show_history":
            self.canvas.show_history = state
        elif name == "show_force_vectors":
            self.canvas.show_force_vectors = state
        elif name == "show_orbital_ellipse":
           self.canvas.show_orbital_ellipse = state
        elif name == "track_focused":
           self.canvas.track_focused = state
        elif name == "show_magnifier":
           self.canvas.show_magnifier = state

    def __build_menu(self) -> Gio.Menu:
        def bool_item(label: str | None = None, detailed_action: str | None = None):
            item = Gio.MenuItem.new(label, detailed_action)
            item.set_attribute_value("toggle", GLib.Variant.new_boolean(True))
            return item
        
        menu = Gio.Menu()
        menu.append_item(bool_item("Show grid",    "app.show_grid"))
        menu.append_item(bool_item("Show history", "app.show_history"))
        
        focused_menu = Gio.Menu()
        # focused_menu.append_item(bool_item("Show force vectors", "app.show_force_vectors"))
        focused_menu.append_item(bool_item("Track focused", "app.track_focused"))
        focused_menu.append_item(bool_item("Show focused magnifier", "app.show_magnifier"))
        focused_menu.append_item(bool_item("Show orbit", "app.show_orbital_ellipse"))
        
        section = Gio.Menu()
        section.append("Quit", "app.quit")
        
        top = Gio.Menu()
        top.append_section("Options", menu)
        top.append_section("Focused", focused_menu)
        top.append_section(None, section)
        
        return top

    def do_activate(self):
        Gtk.Application.do_activate(self)
        win = self.props.active_window 
        if not win:
            win = self._build_window()
            win.set_default_size(*WINDOW_DEFAULT_SIZE)
            win.set_child(self.canvas)
            #win.add_tick_callback(self.on_tick)
        win.present()
        if self.start_maximized:
            win.maximize()
    
    def _build_window(self) -> Gtk.ApplicationWindow:
        win = Gtk.ApplicationWindow(application=self, title=APP_ID)
        win.set_default_size(*WINDOW_DEFAULT_SIZE)
        win.set_child(self.canvas)
        
        # Key controller
        controller = Gtk.EventControllerKey.new()
        controller.connect("key-pressed", self.on_key_pressed)
        controller.connect("key-released", self.on_key_released)
        win.add_controller(controller)
        
        header = Gtk.HeaderBar()
        
        # Menu button wired to our Gio.MenuModel
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic")
        menu_button.set_menu_model(self.__build_menu())
        header.pack_end(menu_button)

        # Adjustment controls range/step/value
        adj = Gtk.Adjustment(
            value=1,
            lower=0.1,
            upper=100,
            step_increment=0.1,
            page_increment=10
        )
        
        self.scale_speed = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj, can_focus=False, focusable=False)
        self.scale_speed.set_vexpand(True)
        self.scale_speed.set_size_request(250, 25)      #  width, height
        self.scale_speed.set_value_pos(Gtk.PositionType.RIGHT)
        #for mark in self._tick_rate_opts:
        #    self.scale_speed.add_mark(self._tick_rate_opts.index(mark)+1, Gtk.PositionType.RIGHT, None)
        header.pack_start(self.scale_speed)
        
        lbl_speed = Gtk.Label(label=f"{self.canvas.orbital.speed:4.1f}x")
        lbl_speed.set_size_request(60, 25)
        header.pack_start(lbl_speed)
        win.set_titlebar(header)
        
        def on_speed_slider_value_changed(scale: Gtk.Scale):
            value = scale.get_value()
            self.canvas.orbital.speed = value
            lbl_speed.set_text(f"{value:4.1f}x")
        self.scale_speed.connect("value-changed", on_speed_slider_value_changed)
        return win

    def on_escape(self):
        if self.canvas.secondary_idx is not None:
            self.canvas.secondary_idx = None
            return True
        self.quit()
        return True
 
    def on_wasd(self, ctrl_held):
        if not self.canvas.secondary_idx:
            return
        MAX_THRUST = 10.0
        ax, ay = 0, 0
        if Gdk.KEY_w in self.pressed_keys:
            ay -= MAX_THRUST
        if Gdk.KEY_a in self.pressed_keys:
            ax -= MAX_THRUST
        if Gdk.KEY_s in self.pressed_keys:
            ay += MAX_THRUST
        if Gdk.KEY_d in self.pressed_keys:
            ax += MAX_THRUST
        self.canvas.apply_accel(self.canvas.secondary_idx, ax, ay)

    def on_key_pressed(self, controller: Gtk.EventControllerKey, keyval: int, keycode: int, state: Gdk.ModifierType) -> bool:
        WASD_KEYS = [Gdk.KEY_w, Gdk.KEY_a, Gdk.KEY_s, Gdk.KEY_d]
        self.pressed_keys.add(keyval)
        ctrl_held = state & Gdk.ModifierType.CONTROL_MASK
        mod_held = state & Gdk.ModifierType.ALT_MASK
        
        if keyval == Gdk.KEY_Escape:
            self.on_escape()
        elif keyval in [Gdk.KEY_Tab, Gdk.KEY_ISO_Left_Tab] and self.canvas.secondary_idx is not None:    
            valid_indices:list = np.unique(self.canvas.orbital.ix).tolist()
            valid_indices.remove(self.canvas.primary_idx)
            try:
                idx_current = valid_indices.index(self.canvas.secondary_idx)
                idx_new = idx_current + (-1 if keyval == Gdk.KEY_ISO_Left_Tab else 1)
                if idx_new >= len(valid_indices):
                    self.canvas.secondary_idx = valid_indices[0]
                elif idx_new < 0:
                    self.canvas.secondary_idx = valid_indices[-1]
                else:
                    self.canvas.secondary_idx = valid_indices[idx_new]
            except ValueError:
                self.canvas.secondary_idx = valid_indices[0]
        
        elif keyval in WASD_KEYS:
            self.on_wasd(ctrl_held)
        
        elif keyval == Gdk.KEY_Left:
            self.scale_speed.set_value(self.scale_speed.get_value() - 0.1)
        
        elif keyval == Gdk.KEY_Right:
           self.scale_speed.set_value(self.scale_speed.get_value() + 0.1)
        
        elif keyval == Gdk.KEY_Down:
            self.canvas.zoom_in()
        
        elif keyval == Gdk.KEY_Up:
            self.canvas.zoom_out()
        
        elif keyval == Gdk.KEY_space:
            self.paused = not self.paused
            self.canvas.paused = not self.canvas.paused
        
        elif keyval == Gdk.KEY_f:
            win = self.get_active_window()
            if win is None:
                return False
            if mod_held:
                if win.is_maximized():
                    win.unmaximize()
                else:
                    win.maximize()
            else:
                if win.is_fullscreen():
                    win.unfullscreen()
                else:
                    win.fullscreen()
        return False

    def on_key_released(self, controller, keyval, keycode, state):
        self.pressed_keys.discard(keyval)
        return False
    
    def on_simulation_tick(self, widget, frame_clock:Gdk.FrameClock|None):
        if self.paused:
            return True
        
        start = time.perf_counter()
        
        # (Temporary)
        # Lock the speed to the computed max speed
        if frame_clock is not None and frame_clock.get_fps() and len(self.canvas.t_render) > 0 and len(self.orbital.t_step) > 0:
            t_render = statistics.mean(self.canvas.t_render)
            t_step = statistics.mean(self.orbital.t_step)
            
            #fps = frame_clock.get_fps()
            #frame_time = 1/fps
            #margin = frame_time * 0.2
            #speed_max = ((frame_time - margin - t_render) / t_step) * (self.orbital.dt_base / frame_time)
            #self.scale_speed.set_value(speed_max)
            
            self.canvas.avg_step_duration = t_step
            self.canvas.avg_render_duration = t_render
        
        steps = self.orbital.frame(start)

        self.tick += 1
        self._tick_steps.append(steps)
        self._tick_duration.append(time.perf_counter() - start)
        
        self.canvas.frame_clock = frame_clock
        return True

    def compute_history(self):
        for idx in range(self.orbital.shm.mass.size):
            if self.orbital.shm.mass[idx] == 0:
                continue
            self.canvas.compute_history(idx)
            
    def idle_loop(self):
        win = self.get_active_window()
        if win is None:
            return False
        
        clock = win.get_frame_clock()
        self.on_simulation_tick(self.canvas, clock)
        GLib.idle_add(self.compute_history)
        GLib.idle_add(self.idle_loop)
        