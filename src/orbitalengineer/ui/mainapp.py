import json
from typing import cast

from orbitalengineer.engine.particle import Particle
from orbitalengineer.ui.select_device_window import SelectDeviceWindow

from orbitalengineer.helpers import seed
from orbitalengineer.names import make_name

from orbitalengineer.ui import model, ui_config
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.ticker import TickController
from orbitalengineer.ui.mainwindow import MainWindow
from orbitalengineer.ui.gtk4 import Gtk, Gio, GLib, GObject
from orbitalengineer.ui.keyinput import KeyInput

from orbitalengineer.engine import config, log_timing, logger
from orbitalengineer.engine.orbitalcl.orbitalcl import SimController_CL
from orbitalengineer.engine.clock import SimClock


class App(Gtk.Application):
    
    platform_id = GObject.Property(type=int, default=-1)
    device_id = GObject.Property(type=int, default=-1)
    
    def __init__(self, resume_from_file:str|bool=False, platform_id=-1, device_id=-1):
        super().__init__(application_id=ui_config.APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.resume_from_file = resume_from_file
        
        self.view = model.ViewModel()
        self.clock = SimClock()
        self.orbital = SimController_CL()
        self.camera = pz.Camera2D()
        
        self.view.connect("notify::paused", self.on_paused_changed)
        self.view.connect("notify::speed", self.on_speed_changed)
        self.view.props.paused = True
        self.platform_id = platform_id
        self.device_id = device_id

    def on_paused_changed(self, model, param):
        self._toggle_paused()
    
    def on_speed_changed(self, model, param):
        self.clock.speed = self.view.props.speed

    def start_tick(self):
        self.orbital.accum = 0
        self.orbital.last_now = None
        self.clock.start()
        self.tick_ctl = TickController(self.orbital, self.clock)
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
        self.orbital.accum = 0
        self.orbital.last_now = None
        self.clock.increment_by (self.orbital.dt_base)
        if self.orbital.tick(self.clock.time()) > 0:
            GLib.idle_add(self.orbital.sync)

    def insert_particle(self, particle:Particle, color:tuple[float, float, float, float]=(1,1,1,1)) -> int:
        idx = self.orbital.add_particle(particle)
        self.view.particle_colors[idx] = random_color() if color is None else color
        self.view.particle_names[idx] = make_name(seed + idx)
        return idx
    
    def _on_close_request(self, dialog: SelectDeviceWindow):
        selection = dialog.get_selection()
        if selection is None:
            self.quit()
            return
        platform_id, device_id = selection
        self.platform_id = platform_id
        self.device_id = device_id
        self.orbital.set_cl_device(platform_id, device_id)
        self.orbital.init_sim()
        self.init_mainwindow()
    
    def init_mainwindow(self) -> MainWindow:
        win = MainWindow(
            application=self,
            title=ui_config.DEFAULT_WINDOW_TITLE,
            camera=self.camera,
            view=self.view,
            ctl=self.orbital,
            clock=self.clock,
        )
        self.key_input = KeyInput(self, win)
        win.present()
        if self.view.start_maximized: win.maximize()
        return win
    
    def do_activate(self):
        if self.resume_from_file:
            self.load_from_file()
        if self.platform_id == -1 or self.device_id == -1:
            dialog = SelectDeviceWindow()
            dialog.set_application(self)
            dialog.connect("close-request", self._on_close_request)
            dialog.present()
        else:
            self.orbital.set_cl_device(self.platform_id, self.device_id)
            self.orbital.init_sim()
            self.init_mainwindow()


    def shift_focus(self, particle_id):
        b = self.orbital.get_particle(particle_id)
        if b is None:
            logger.warning(f"Unknown focus: {particle_id}")
            return
        self.view.secondary_body = particle_id

        win = self.props.active_window
        if not win:
            available_size = 300
        else:
            available_size = min(win.get_allocated_height(), win.get_allocated_width()) * 0.25

        radius = b.get_radius()
        diameter = 3 * radius
        
        #if self.view.follow_tracked_body:
        #    if diameter * self.camera.zoom > available_size:
        #        self.camera.zoom = available_size / diameter
        #    elif diameter * self.camera.zoom < 10:
        #        self.camera.zoom = 10 / diameter

    def relative_zoom(self, factor):
        self.camera.zoom_at(0, 0, 0, 0, factor)

    # def on_collision(self, event:BouncingCollisionEvent):
    #     r1 = self.orbital.get_particle(event.i).get_radius()
    #     r2 = self.orbital.get_particle(event.j).get_radius()
    #     size = min(r1, r2)/2.0
    #     self.view.pinpoint.append(model.Pinpoint(
    #         position=event.collision_point,
    #         start=self.clock.time(),
    #         until=self.clock.time() + 0.1,
    #         radius=size * self.camera.zoom
    #     ))

    def to_dict(self) -> dict:
        return {
            "clock": self.clock.to_dict(),
            "camera": self.camera.to_dict(),
            "orbital": self.orbital.to_dict(),
            "view": self.view.to_dict(),
        }
    
    def load_from_file(self):
        if self.resume_from_file is False: return

        file_name = ui_config.DEFAULT_SCENARIO_FILE
        if isinstance(self.resume_from_file, str):
            file_name = self.resume_from_file
        logger.info(f"Loading from {file_name}")
        obj = cast(dict, json.load(open(file_name, "r")))
        
        self.platform_id = obj.get('platform_id', self.platform_id)
        self.device_id = obj.get('device_id', self.device_id)
        
        self.clock = SimClock()
        self.clock._speed = obj["clock"]["_speed"]
        self.clock._duration = obj["clock"]["_duration"]
        self.clock.running = obj["clock"]["running"]
        
        self.camera.zoom = obj["camera"]["zoom"]
        self.camera.offset = obj["camera"]["offset"]
        
        self.view.load_from_dict(obj["view"])
        self.orbital.load_from_dict(obj["orbital"])
    
    def save_scenario(self):
        if self.orbital.is_initialized:
            logger.info(f"Saving to {ui_config.DEFAULT_SCENARIO_FILE}")
            json.dump(self.to_dict(), open(ui_config.DEFAULT_SCENARIO_FILE, "w"))