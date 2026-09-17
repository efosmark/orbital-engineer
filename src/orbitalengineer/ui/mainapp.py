from orbitalengineer.ui import ui_config
from orbitalengineer.ui.audio.synth import ToneSynthController
from orbitalengineer.ui.client_sync import ClientSyncController
from orbitalengineer.ui.model import main
from orbitalengineer.ui.select_device_window import SelectDeviceWindow
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.mainwindow import MainWindow
from orbitalengineer.ui.gtk4 import Gtk, Gio, GObject, GLib
from orbitalengineer.ui.keyinput import KeyInput

from orbitalengineer.engine import logger
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ipc.client import ClientSocketConnection


class App(Gtk.Application):
    platform_id = GObject.Property(type=int, default=-1)
    device_id = GObject.Property(type=int, default=-1)
    
    def __init__(self, platform_id=-1, device_id=-1):
        super().__init__(application_id=ui_config.APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)        

        self.model = main.AppModel()
        self.client = ClientSocketConnection()
        self.sync_ctl = ClientSyncController(self.client, self.model.engine)
        
        self.camera = pz.Camera2D()
        self.synth = ToneSynthController()
        
        self.model.engine.connect("notify::paused", self.on_paused_changed)
        self.model.engine.connect("notify::clock-speed", self.on_speed_changed)
        self.model.engine.connect("notify::max-speed", self.on_max_speed_changed)
        self.model.connect("notify::show-grid", self.on_show_grid_changed)
        self.model.cmap.connect('notify::cmap', self.on_color_map_changed)
        self.model.engine.paused = True
        self.platform_id = platform_id
        self.device_id = device_id
        self.client.set_device(platform_id, device_id)

    def bootstrap(self, reset:bool=True):
        self.client.connect(reset=reset)
        self.client.sync_full_state()
        self.model.osd.add_message("Ready.", duration=10.0, desc="Press [space] to start.")

    def on_color_map_changed(self, model, param):
        self.model.osd.add_message(f"Color Map: {self.model.cmap.cmap}")

    def on_max_speed_changed(self, model, param):
        #if self.view.max_speed < self.view.speed:
        #    self.view.props.speed = self.view.max_speed
        ...

    def on_show_grid_changed(self, model, param):
        grid_state = "On" if self.model.show_grid else "Off"
        self.model.osd.add_message(f"Grid: {grid_state}")

    def on_paused_changed(self, model, param):
        self._toggle_paused()
        if self.model.engine.paused:
            if self.model.engine.tick_id > 0:
                self.model.osd.add_message("Paused", duration=3.0)
        else:
            self.model.osd.add_message("Running")
    
    def on_speed_changed(self, model, param):
        speed = self.model.engine.clock_speed
        precision = 1 if speed >= 0.1 else 2
        f_cur_speed = f"{speed:.{precision}f}"
        self.model.osd.add_message(f"Speed: {f_cur_speed}x")
    
    def _toggle_paused(self):
        if self.model.engine.paused:
            self.client.stop()
        else:
            self.client.start()

    def insert_particle(self, particle:Particle, color:tuple[float, float, float, float]=(1,1,1,1)) -> int:
        idx = self.client.add_particle(
            position=particle.get_position(),
            velocity=particle.get_velocity(),
            mass=particle.get_mass(),
            radius=particle.get_radius(),
            flags=particle.get_flags()
        )
        self.model.particle_colors[idx] = random_color() if color is None else color
        self.model.particle_names[idx] = f"body-{idx}"
        return idx
    
    def init_mainwindow(self) -> MainWindow:
        win = MainWindow(
            application=self,
            title=ui_config.DEFAULT_WINDOW_TITLE,
            camera=self.camera,
            view=self.model,
            ctl=self.client,
            clock=self.client.clock,
            synth=self.synth
        )
        self.key_input = KeyInput(self, win)
        win.present()
        return win
    
    def select_device(self):
        def _on_close_device_selection(dialog: SelectDeviceWindow):
            selection = dialog.get_selection()
            if selection is None:
                return self.quit()
            self.client.set_device(selection[0], selection[1])
            self.init_mainwindow()
        dialog = SelectDeviceWindow()
        dialog.set_application(self)
        dialog.connect("close-request", _on_close_device_selection)
        dialog.present()
    
    def do_activate(self):
        if self.platform_id == -1 or self.device_id == -1:
            self.select_device()
        else:
            self.client.set_device(self.platform_id, self.device_id)
            self.init_mainwindow()

    def shift_focus(self, particle_id):
        b = self.client.get_particle(particle_id)
        if b is None:
            logger.warning(f"Unknown focus: {particle_id}")
            return
        self.model.secondary_body = particle_id
        self.model.osd.add_message(f"Focus: {particle_id}", duration=0.5)

        # win = self.props.active_window
        # if not win:
        #     available_size = 300
        # else:
        #     available_size = min(win.get_allocated_height(), win.get_allocated_width()) * 0.25
        # radius = b.get_radius()
        # diameter = 3 * radius
        #if self.view.follow_tracked_body:
        #    if diameter * self.camera.zoom > available_size:
        #        self.camera.zoom = available_size / diameter
        #    elif diameter * self.camera.zoom < 10:
        #        self.camera.zoom = 10 / diameter

    def tick_once(self):
        if not hasattr(self, '_last_tick'):
            self._last_tick = None
        if self._last_tick is None or self.client.tick_id > self._last_tick:
            self._last_tick = self.client.tick_id
            self.model.osd.add_message("Tick", duration=0.25)
            GLib.idle_add(self.client.tick_once)

    def substep_once(self):
        self.model.osd.add_message("Sub-step", duration=0.25)
        self.client.substep_once()
    
    def relative_zoom(self, factor):
        self.camera.zoom_at(0, 0, 0, 0, factor)

    # def on_collision(self, event:BouncingCollisionEvent):
    #     r1 = self.client.get_particle(event.i).get_radius()
    #     r2 = self.client.get_particle(event.j).get_radius()
    #     size = min(r1, r2)/2.0
    #     self.view.pinpoint.append(model.Pinpoint(
    #         position=event.collision_point,
    #         start=self.clock.time(),
    #         until=self.clock.time() + 0.1,
    #         radius=size * self.camera.zoom
    #     ))

    # def to_dict(self) -> dict:
    #     return {
    #         "camera": self.camera.to_dict(),
    #         "orbital": self.client.to_dict(),
    #         "view": self.view.to_dict(),
    #     }
    
    # def load_from_file(self):
    #     if self.resume_from_file is False: return

    #     file_name = ui_config.DEFAULT_SCENARIO_FILE
    #     if isinstance(self.resume_from_file, str):
    #         file_name = self.resume_from_file
    #     logger.info(f"Loading from {file_name}")
    #     obj = cast(dict, json.load(open(file_name, "r")))
        
    #     self.platform_id = obj.get('platform_id', self.platform_id)
    #     self.device_id = obj.get('device_id', self.device_id)
        
    #     #self.clock = SimClock()
    #     #self.clock.speed = obj["clock"]["speed"]
    #     #self.clock._duration = obj["clock"]["duration"]
    #     #self.clock.running = obj["clock"]["running"]
        
    #     self.camera.zoom = obj["camera"]["zoom"]
    #     self.camera.offset = obj["camera"]["offset"]
        
    #     self.view.load_from_dict(obj["view"])
    #     self.orbital.load_from_dict(obj)
    
    # def save_scenario(self):
    #     if self.client.is_initialized:
    #         logger.info(f"Saving to {ui_config.DEFAULT_SCENARIO_FILE}")
    #         json.dump(self.to_dict(), open(ui_config.DEFAULT_SCENARIO_FILE, "w"))