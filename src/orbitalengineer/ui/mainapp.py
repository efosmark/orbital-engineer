from typing import cast
import numpy as np

from orbitalengineer.ui.select_device_window import SelectDeviceWindow
np.set_printoptions(precision=3)

from orbitalengineer.helpers import seed
from orbitalengineer.names import make_name

from orbitalengineer.ui import model
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.ticker import TickController
from orbitalengineer.ui.mainwindow import MainWindow
from orbitalengineer.ui.gtk4 import Gtk, Gio, GLib
from orbitalengineer.ui.keyinput import KeyInput

from orbitalengineer.engine import logger
from orbitalengineer.engine.cl.orbitalcl import SimController_CL
from orbitalengineer.engine.simcontroller import Particle
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.event import BouncingCollisionEvent


APP_ID = "com.qmew.OrbitalEngineer-dialog"
WINDOW_DEFAULT_SIZE = (1200, 800)

class App(Gtk.Application):
    
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.view = model.ViewModel()
        self.data = model.DataModel()
        self.clock = SimClock()
        
        self.orbit_ctl = SimController_CL(self.clock)
        self.orbit_ctl.on_collision = self.on_collision

        self.camera = pz.Camera2D()
        
        self.view.connect("notify::paused", self.on_paused_changed)
        self.view.connect("notify::speed", self.on_speed_changed)
        self.view.props.paused = True

    def on_paused_changed(self, model, param):
        self._toggle_paused()
    
    def on_speed_changed(self, model, param):
        self.orbit_ctl.clock.set_speed(self.view.props.speed)

    def start_tick(self):
        self.orbit_ctl.accum = 0
        self.orbit_ctl.last_now = None
        self.clock.start()
        self.tick_ctl = TickController(self.data, self.orbit_ctl, self.clock)
        self.tick_ctl.start()
        
    def _toggle_paused(self):
        if self.view.props.paused:
            self.clock.stop()
            if hasattr(self, 'tick_ctl'):
                self.tick_ctl.stop()
        else:
            self.start_tick()

    def tick_once(self):
        if not self.view.props.paused: return
        self.orbit_ctl.accum = 0
        self.orbit_ctl.last_now = None
        self.clock.increment_by(self.orbit_ctl.dt_base)
        if self.orbit_ctl.tick(self.clock.time()) > 0:
            GLib.idle_add(self.orbit_ctl.sync)

    def insert_particle(self, particle:Particle, color:tuple[float, float, float, float]=(1,1,1,1)) -> int:
        idx = self.orbit_ctl.add_particle(particle)
        self.view.particle_colors[idx] = random_color() if color is None else color
        self.view.particle_names[idx] = make_name(seed + idx)
        return idx
    
    def do_startup(self):
        Gtk.Application.do_startup(self)

    def _on_close_request(self, dialog: SelectDeviceWindow):
        selection = dialog.get_selection()
        if selection is None:
            self.quit()
            return
        self.orbit_ctl.set_cl_device(*selection)
        self.orbit_ctl.init_sim()
        self.view.props.paused = False

    def do_activate(self):
        Gtk.Application.do_activate(self)
        
        win = cast(MainWindow, self.props.active_window)
        if not win:
            win = MainWindow(
                application=self,
                title=APP_ID,
                camera=self.camera,
                view=self.view,
                data=self.data,
                ctl=self.orbit_ctl,
            )
            logger.info("Main window configured and ready to present.")
        self.key_input = KeyInput(self, win)
         
        win.present()
        if self.view.start_maximized:
            win.maximize()
        
        dialog = SelectDeviceWindow(parent=win)
        dialog.set_application(self)
        dialog.connect("close-request", self._on_close_request)
        dialog.present()

    def shift_focus(self, particle_id):
        b = self.orbit_ctl.get_particle(particle_id)
        if b is None:
            logger.warning(f"Unknown focus: {particle_id}")
            return
        self.data.secondary_body = particle_id

        win = self.props.active_window
        if not win:
            available_size = 300
        else:
            available_size = min(win.get_allocated_height(), win.get_allocated_width()) * 0.25

        radius = b.get_radius()
        diameter = 3 * radius
        if diameter * self.camera.zoom > available_size:
            self.camera.zoom = available_size / diameter
        elif diameter * self.camera.zoom < 10:
            self.camera.zoom = 10 / diameter

    def relative_zoom(self, factor):
        self.camera.zoom_at(0, 0, 0, 0, factor)

    def on_collision(self, event:BouncingCollisionEvent):
        r1 = self.orbit_ctl.get_particle(event.i).get_radius()
        r2 = self.orbit_ctl.get_particle(event.j).get_radius()
        
        size = min(r1, r2)/2.0
        self.view.pinpoint.append(model.Pinpoint(
            position=event.collision_point,
            start=self.clock.time(),
            until=self.clock.time() + 0.1,
            radius=size * self.camera.zoom
        ))