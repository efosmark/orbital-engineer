import numpy as np
np.set_printoptions(precision=3)

from orbitalengineer.helpers import seed
from orbitalengineer.names import make_name
from orbitalengineer.ui import model
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.ticker import TickController
from orbitalengineer.ui.mainwindow import MainWindow
from orbitalengineer.ui.fmt import get_scale_value
from orbitalengineer.ui.gtk4 import Gtk, Gio, GLib
from orbitalengineer.ui.plot import PlotWindow
from orbitalengineer.ui.keyinput import KeyInput

from orbitalengineer.engine.cl.orbitalcl import SimController_CL
from orbitalengineer.engine.simcontroller import Particle
from orbitalengineer.engine import logger


APP_ID = "com.qmew.OrbitalEngineer-dialog"
WINDOW_DEFAULT_SIZE = (1200, 800)

class App(Gtk.Application):

    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.view = model.ViewModel()
        self.data = model.DataModel()
        
        self.orbit_ctl = SimController_CL()
        self.ticker = TickController(self.data, self.orbit_ctl)
        self.camera = pz.Camera2D()

    
    def insert_particle(self, particle:Particle, color:tuple[float, float, float, float]=(1,1,1,1)) -> int:
        idx = self.orbit_ctl.add_particle(particle)
        self.view.particle_colors[idx] = random_color() if color is None else color
        self.view.particle_names[idx] = make_name(seed + idx)
        return idx
    
    def do_startup(self):
        Gtk.Application.do_startup(self)

    def do_activate(self):
        Gtk.Application.do_activate(self)
        win = self.props.active_window
        
        if not win:
            win = MainWindow(
                application=self,
                title=APP_ID,
                camera=self.camera,
                view=self.view,
                data=self.data,
                ctl=self.orbit_ctl,
            )
            self.key_input = KeyInput(self, win)
            logger.info("Main window configured and ready to present.")
        
        win.present()
        if self.view.start_maximized:
           win.maximize()
        
        self.plot_win = None
        if self.view.show_plot_at_startup:
            self.plot_win = PlotWindow(
                self,
                self.data
            )
            self.plot_win.present()

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
        diameter = 2 * radius
        if diameter * self.camera.zoom > available_size:
            self.camera.zoom = available_size / diameter
        elif diameter * self.camera.zoom < 10:
            self.camera.zoom = 10 / diameter

    def relative_zoom(self, factor):
        self.camera.zoom_at(0, 0, 0, 0, factor)
        z = get_scale_value(self.camera.zoom)
        self.view.add_osd_message(f"Zoom: {z:.1f}X", 0.75)
