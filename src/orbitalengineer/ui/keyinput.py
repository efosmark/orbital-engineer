import time
from typing import Any

import numpy as np
np.set_printoptions(precision=3)

from orbitalengineer.ui.model import Pinpoint
from orbitalengineer.engine.cl.orbitalcl import SimController_CL
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
            self.on_wasd()
        elif keyval == Gdk.KEY_Left:
            self.app.orbit_ctl.speed -= 0.1
        elif keyval == Gdk.KEY_Right:
            self.app.orbit_ctl.speed += 0.1
        elif keyval == Gdk.KEY_Down:
            self.app.relative_zoom(0.9)
        elif keyval == Gdk.KEY_Up:
            self.app.relative_zoom(1 / 0.9)
        #elif keyval == Gdk.KEY_p:
        #    self.app.paused = not self.app.paused
        elif keyval == Gdk.KEY_f:
            self.toggle_screen_size()
        #elif keyval == Gdk.KEY_Delete and self.app.data.secondary_body is not None:
        #    self.app.shm.status[self.app.data.secondary_body] = STATUS_DELETED
        return False

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
    
    def on_wasd(self):
        if not self.app.data.secondary_body:
            return
        MAX_THRUST = 0.15
        ax, ay = 0, 0
        if Gdk.KEY_w in self.pressed_keys:
            ay -= MAX_THRUST
        if Gdk.KEY_a in self.pressed_keys:
            ax -= MAX_THRUST
        if Gdk.KEY_s in self.pressed_keys:
            ay += MAX_THRUST
        if Gdk.KEY_d in self.pressed_keys:
            ax += MAX_THRUST
        self.app.orbit_ctl.inout_queue.append((self.app.data.secondary_body, ax, ay))
        
        A = self.app.orbit_ctl.get_particle(self.app.data.secondary_body)
        start = time.monotonic()
        
        accel_angle = np.atan2(ay, ax) - np.pi
        radius = A.get_radius()
        
        dist = radius/2
        size = radius/2
        for i in range(1, 21):
            k = 1 - (i / 10.0)
            
            size = radius * k
            dist += size
            x = dist * np.cos(accel_angle)
            y = dist * np.sin(accel_angle)

            self.app.view.pinpoint.append(Pinpoint(
                position=A.get_position() + complex(x, y),
                start=start,
                until=start + k/4,
                radius=size * self.app.camera.zoom
            ))
    
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
        if win is None:
            return False
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
