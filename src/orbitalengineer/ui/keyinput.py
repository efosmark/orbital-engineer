import cmath
from typing import Any, cast
from orbitalengineer.ui import model
from orbitalengineer.ui.gtk4 import Gtk, Gdk, GObject


class KeyInput(GObject.GObject):
    WASD_KEYS = [Gdk.KEY_w, Gdk.KEY_a, Gdk.KEY_s, Gdk.KEY_d]
    TAB_NEXT = Gdk.KEY_Tab
    TAB_PREV = Gdk.KEY_ISO_Left_Tab
    
    props:Any
    
    pressed_keys = GObject.Property(type=object)
    ctrl_held = GObject.Property(type=bool, default=False)
    mod_held = GObject.Property(type=bool, default=False)
    
    def __init__(self, app, win:Gtk.ApplicationWindow):
        super().__init__()
        self.mainapp = app
        self.app = cast(model.AppModel, self.mainapp.model)
        self.props.pressed_keys = set()

        controller = Gtk.EventControllerKey.new()
        controller.connect("key-pressed", self.on_key_pressed)
        controller.connect("key-released", self.on_key_released)
        win.add_controller(controller)

    def on_key_pressed(self, controller: Gtk.EventControllerKey, keyval: int, keycode: int, state: Gdk.ModifierType) -> bool:
        self.props.pressed_keys.add(keyval)
        self.props.ctrl_held = bool(state & Gdk.ModifierType.CONTROL_MASK)
        self.props.mod_held = bool(state & Gdk.ModifierType.ALT_MASK)
        self.notify('pressed_keys')

        if keyval == Gdk.KEY_Escape:
            self.on_escape()
        elif keyval == Gdk.KEY_l:
            self.app.cycle_color_map(self.props.ctrl_held)
        elif keyval in [self.TAB_PREV, self.TAB_NEXT] and self.app.secondary_body is not None:    
            direction = -1 if keyval == self.TAB_PREV else 1
            self.cycle_particles(direction)
        # elif keyval == Gdk.KEY_s and self.props.ctrl_held:
        #     self.app.save_scenario()
        elif keyval in self.WASD_KEYS:
            self.on_wasd(keyval)
        elif keyval == Gdk.KEY_Left:
            self.app.engine.decrease_speed()
        elif keyval == Gdk.KEY_Right:
            self.app.engine.increase_speed()
        elif keyval == Gdk.KEY_Down:
            self.mainapp.relative_zoom(0.9)
        elif keyval == Gdk.KEY_Up:
            self.mainapp.relative_zoom(1 / 0.9)
        elif keyval == Gdk.KEY_space:
            self.app.engine.paused = not self.app.engine.paused
        elif keyval == Gdk.KEY_f:
            self.toggle_screen_size()
        elif keyval == Gdk.KEY_period:
            if self.props.ctrl_held:
                self.mainapp.substep_once()
            else:
                self.mainapp.tick_once()
        return False

    def on_wasd(self, keyval):
        
        # Temporary test controls for apply_vector_offset
        if self.props.mod_held:
            if keyval == Gdk.KEY_w:
                self.mainapp.client.rel_mass(self.app.selected_particles, 1.1)
            elif keyval == Gdk.KEY_s:
                self.mainapp.client.rel_mass(self.app.selected_particles, 1/1.1)
            elif keyval == Gdk.KEY_a:
                self.mainapp.client.rel_velocity(self.app.selected_particles, 1.1)
            elif keyval == Gdk.KEY_d:
                self.mainapp.client.rel_velocity(self.app.selected_particles, 1/1.1)
            return
        
        if self.app.secondary_body is None: return
        b = self.mainapp.client.get_particle(self.app.secondary_body)
        r, angle = cmath.polar(b.get_velocity())
        if keyval == Gdk.KEY_a:
            angle -= ((2*cmath.pi) / 360.0)
            velocity = cmath.rect(r, angle)
        elif keyval == Gdk.KEY_d:
            angle += ((2*cmath.pi) / 360.0)
            velocity = cmath.rect(r, angle)
        elif keyval == Gdk.KEY_w:
            r, angle = cmath.polar(b.get_velocity())
            velocity = cmath.rect(r * 1.005, angle)
        elif keyval == Gdk.KEY_s:
            r, angle = cmath.polar(b.get_velocity())
            velocity = cmath.rect(r / 1.005, angle)
        else:
            return
        self.mainapp.client.set_velocity(self.app.secondary_body, velocity.real, velocity.imag)

    def on_escape(self):
        if self.app.secondary_body is not None:
            self.app.secondary_body = None
            return True
        self.mainapp.quit()
        return True
 
    def on_key_released(self, controller, keyval, keycode, state):
        self.props.pressed_keys.discard(keyval)
        self.notify('pressed_keys')
        return False

    def cycle_particles(self, direction:int):
        valid_indices:list = self.mainapp.client.get_valid_indices().tolist()
        try:
            idx_current = valid_indices.index(self.app.secondary_body)
            idx_new = idx_current + direction
            if idx_new >= len(valid_indices):
                self.mainapp.shift_focus(valid_indices[0])
            elif idx_new < 0:
                self.mainapp.shift_focus(valid_indices[-1])
            else:
                self.mainapp.shift_focus(valid_indices[idx_new])
        except ValueError:
            self.mainapp.shift_focus(valid_indices[0])

    def toggle_screen_size(self):
        # TODO: Notify the app and have it handle this rather than performing it here
        win = self.mainapp.get_active_window()
        if win is None: return False
        if self.mod_held:
            if win.is_maximized():
                win.unmaximize()
            else:
                win.maximize()
        else:
            if win.is_fullscreen():
                win.unfullscreen()
            else:
                win.fullscreen()
