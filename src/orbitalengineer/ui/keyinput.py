import cmath
from typing import Any
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
        self.app = app
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
        elif keyval in [self.TAB_PREV, self.TAB_NEXT] and self.app.data.secondary_body is not None:    
            direction = -1 if keyval == self.TAB_PREV else 1
            self.cycle_particles(direction)
        elif keyval in self.WASD_KEYS:
            self.on_wasd(keyval)
        elif keyval == Gdk.KEY_Left:
            self.app.orbit_ctl.speed -= 0.1
        elif keyval == Gdk.KEY_Right:
            self.app.orbit_ctl.speed += 0.1
        elif keyval == Gdk.KEY_Down:
            self.app.relative_zoom(0.9)
        elif keyval == Gdk.KEY_Up:
            self.app.relative_zoom(1 / 0.9)
        elif keyval == Gdk.KEY_space:
            self.app.view.props.paused = not self.app.view.props.paused
        elif keyval == Gdk.KEY_f:
            self.toggle_screen_size()
        elif keyval == Gdk.KEY_period:
            self.app.tick_once()
        return False

    def on_wasd(self, keyval):
        if self.app.data.secondary_body is None: return
        b = self.app.orbit_ctl.get_particle(self.app.data.secondary_body)
        r, angle = cmath.polar(b.get_velocity())
        
        if keyval == Gdk.KEY_a:
            angle -= (2*cmath.pi / 360.0)
            velocity = cmath.rect(r, angle)
        elif keyval == Gdk.KEY_d:
            angle += (2*cmath.pi / 360.0)
            velocity = cmath.rect(r, angle)
        elif keyval == Gdk.KEY_w:
            r, angle = cmath.polar(b.get_velocity())
            velocity = cmath.rect(r * 1.005, angle)
        elif keyval == Gdk.KEY_s:
            r, angle = cmath.polar(b.get_velocity())
            velocity = cmath.rect(r / 1.005, angle)
        else:
            return
        self.app.orbit_ctl.set_velocity(self.app.data.secondary_body, velocity.real, velocity.imag)

    def on_escape(self):
        if self.app.data.secondary_body is not None:
            self.app.data.secondary_body = None
            return True
        self.app.quit()
        return True
 
    def on_key_released(self, controller, keyval, keycode, state):
        self.props.pressed_keys.discard(keyval)
        self.notify('pressed_keys')
        return False

    def cycle_particles(self, direction:int):
        valid_indices:list = self.app.orbit_ctl.get_valid_indices().tolist()
        try:
            idx_current = valid_indices.index(self.app.data.secondary_body)
            idx_new = idx_current + direction
            if idx_new >= len(valid_indices):
                self.app.shift_focus(valid_indices[0])
            elif idx_new < 0:
                self.app.shift_focus(valid_indices[-1])
            else:
                self.app.shift_focus(valid_indices[idx_new])
        except ValueError:
            self.app.shift_focus(valid_indices[0])

    def toggle_screen_size(self):
        win = self.app.get_active_window()
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
