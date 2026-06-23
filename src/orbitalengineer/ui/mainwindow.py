from typing import Any, cast

from orbitalengineer.engine.orbitalcl.orbitalcl import SimController_CL
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.ui import canvas, ui_config
from orbitalengineer.ui.gtk4 import Gtk, Gio, GLib, GObject
from orbitalengineer.ui.model import ViewModel


class MainWindow(Gtk.ApplicationWindow):

    def __init__(self, application, title, camera, view: ViewModel, ctl:SimController_CL, clock:SimClock):
        Gtk.ApplicationWindow.__init__(self, application=application, title=title)
        self.ctl = ctl
        self.clock = clock
        self._tick_sound = None
        self._tick_audio_ready = False
        
        self.view = view
        
        self.canvas = canvas.OrbitalCanvas(camera, view, self.ctl, self.clock)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        self.canvas.set_halign(Gtk.Align.FILL)
        self.canvas.set_valign(Gtk.Align.FILL)
        
        self.set_default_size(*ui_config.WINDOW_DEFAULT_SIZE)
        self.set_child(self.canvas)

        header = Gtk.HeaderBar()
        self.set_titlebar(header)
        
        # Menu button wired to our Gio.MenuModel
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", can_focus=False)
        menu_button.set_menu_model(self.__build_menu())
        header.pack_end(menu_button)
        self._init_menu_actions()

    def _init_menu_actions(self):
        app = cast(Any, self.get_application())
        if app is None: return

        def add_toggle_action(name: str, initial: bool, handler):
            action = Gio.SimpleAction.new_stateful(
                name, None, GLib.Variant.new_boolean(initial)
            )
            action.connect("change-state", handler)
            app.add_action(action)

        def on_toggle_changed(action: Gio.SimpleAction, value: GLib.Variant):
            action.set_state(value)
            name = action.get_name()
            state = value.get_boolean()
            if name == "show_grid":
                app.view.show_grid = state
            #elif name == "show_history":
                #self.view.show_focused_history = state
            #    app.view.show_all_history = state
            elif name == "show_force_vectors":
                app.view.show_force_vectors = state
            elif name == "show_orbital_ellipse":
                app.view.show_orbital_ellipse = state
            elif name == "track_focused":
                app.view.follow_tracked_body = state
            elif name == "show_magnifier":
                app.view.show_magnifier = state
            elif name == "show_force_vectors":
                app.view.show_force_vectors = state
            elif name == "show_debug_info":
                app.view.show_debug_info = state

        add_toggle_action("show_grid",            app.view.show_grid,            on_toggle_changed)
        add_toggle_action("show_force_vectors",   app.view.show_force_vectors,   on_toggle_changed)
        add_toggle_action("show_history",         app.view.show_focused_history, on_toggle_changed)
        add_toggle_action("show_orbital_ellipse", app.view.show_orbital_ellipse, on_toggle_changed)
        add_toggle_action("track_focused",        app.view.follow_tracked_body,  on_toggle_changed)
        add_toggle_action("show_magnifier",       app.view.show_magnifier,       on_toggle_changed)
        add_toggle_action("show_debug_info",      app.view.show_debug_info,      on_toggle_changed)

        # Quit on the app as well
        act_quit = Gio.SimpleAction.new("quit", None)
        act_quit.connect("activate", lambda a, p: app.quit())
        app.add_action(act_quit)

        # These now match where the actions live
        app.set_accels_for_action("app.quit", ["<Primary>q"])
        app.set_accels_for_action("app.show_grid", ["g"])
        app.set_accels_for_action("app.show_force_vectors", ["v"])
        app.set_accels_for_action("app.show_orbital_ellipse", ["o"])
        app.set_accels_for_action("app.show_history", ["h"])
        app.set_accels_for_action("app.track_focused", ["t"])
        app.set_accels_for_action("app.show_magnifier", ["m"])
        app.set_accels_for_action("app.show_debug_info", ["i"])

    def __build_menu(self) -> Gio.Menu:
        def bool_item(label: str | None = None, detailed_action: str | None = None):
            item = Gio.MenuItem.new(label, detailed_action)
            return item

        menu = Gio.Menu()
        menu.append_item(bool_item("Show grid",    "app.show_grid"))
        menu.append_item(bool_item("Show history", "app.show_history"))
        menu.append_item(bool_item("Show debug",   "app.show_debug_info"))

        focused_menu = Gio.Menu()
        focused_menu.append_item(bool_item("Track focused",        "app.track_focused"))
        focused_menu.append_item(bool_item("Show orbital ellipse", "app.show_orbital_ellipse"))
        focused_menu.append_item(bool_item("Show force vectors",   "app.show_force_vectors"))

        section = Gio.Menu()
        section.append("Quit", "app.quit")

        top = Gio.Menu()
        top.append_section("Options", menu)
        top.append_section("Focused", focused_menu)
        top.append_section(None, section)
        return top
