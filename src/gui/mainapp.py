from orbitalengineer import flags, body
from gui import ui_config, orbitapp
from gui.model import main
from gui.audio.synth import ToneSynthController
from gui.controller.cmap_ctl import ColorizedMapController
from gui.select_device_window import SelectDeviceWindow
from gui.canvas import pz
from gui.mainwindow import MainWindow
from gui.gtk4 import Gtk, Gio, GObject, GLib
from gui.keyinput import KeyInput

from orbitalengineer.engine import logger

class App(orbitapp.OrbitApp):
    
    def __init__(self, platform_id=-1, device_id=-1):
        super().__init__(platform_id, device_id)
        self.model = main.AppModel()
        self.model.engine = self.engine_model
        self.model.ledger = self.ledger_model
        
        self.cmap_ctl = ColorizedMapController(self.model)
        self.camera = pz.Camera2D()
        self.synth = ToneSynthController()
        
        self.engine_model.connect("notify::paused", self.__on_paused_changed)
        self.engine_model.connect("notify::clock-speed", self.on_speed_changed)
        self.model.connect("notify::show-grid", self.on_show_grid_changed)
        self.model.cmap.connect('notify::option', self.on_color_map_changed)
        self.model.cmap.connect('notify::gamma', self.on_color_map_changed)
        self.model.cmap.connect('notify::colormap', self.on_color_map_changed)

    def show_message(self, message:str, duration:float=1, desc:str|None=None):
        self.model.osd.add_message(message, duration=duration, desc=desc)

    def on_color_map_changed(self, model, param):
        mapping_type = self.model.cmap.option
        colormap = self.model.cmap.colormap
        
        desc = None
        if mapping_type is not None:
            desc = f"Color map: {colormap}    Gamma: {self.model.cmap.gamma:.2f}"
        self.model.osd.add_message(f"Showing: {mapping_type.name if mapping_type else 'Normal'}", desc=desc)

    def on_show_grid_changed(self, model, param):
        grid_state = "On" if self.model.show_grid else "Off"
        self.model.osd.add_message(f"Grid: {grid_state}")

    def __on_paused_changed(self, model, param):
        if self.engine_model.tick_id <= 0: return
        if self.engine_model.paused:
            if self.engine_model.tick_id > 0:
                self.model.osd.add_message("Paused", duration=3.0)
        else:
            self.model.osd.add_message("Running")
    
    def on_speed_changed(self, model, param):
        if self.engine_model.tick_id <= 0: return
        speed = self.engine_model.clock_speed
        precision = 1 if speed >= 0.1 else 2
        f_cur_speed = f"{speed:.{precision}f}"
        self.model.osd.add_message(f"Speed: {f_cur_speed}x")

    def insert_particle(self, particle:body.Body, color:tuple[float, float, float, float]=(1,1,1,1)) -> int:
        idx = super().insert_particle(particle)
        self.model.particle_colors[idx] = random_color() if color is None else color
        self.model.particle_names[idx] = f"body-{idx}"
        return idx
    
    def init_mainwindow(self) -> MainWindow:
        win = MainWindow(
            application=self,
            title=ui_config.DEFAULT_WINDOW_TITLE,
            camera=self.camera,
            app_model=self.model,
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
    
    def relative_zoom(self, factor):
        self.camera.zoom_at(0, 0, 0, 0, factor)
