from typing import cast

import numpy as np

from orbitalengineer import flags
from orbitalengineer.ui.model import main
from orbitalengineer.ui.audio.synth import ToneSynthController
from orbitalengineer.ui.canvas.render.cgroup import CGroupConnectionRenderer, CGroupRenderer
from orbitalengineer.ui.canvas.render.gpu_status import GPUStatusRenderer
from orbitalengineer.ui.canvas.render.osd import OSDRenderer
from orbitalengineer.ui.gtk4 import Gtk, Gdk, Graphene
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.canvas.pz import Camera2D, Camera2DController
from orbitalengineer.ui.canvas.render.hud_clock import HudClockRenderer
from orbitalengineer.ui.canvas.render.warning import WarningRenderer
from orbitalengineer.ui.canvas.render.selection import SelectionRenderer
from orbitalengineer.ui.canvas.render.focus_info import FocusInfoRenderer
from orbitalengineer.ui.canvas.render.force import ForceVectorRenderer
from orbitalengineer.ui.canvas.render.debug import DebugInfoRenderer
from orbitalengineer.ui.canvas.render.grid import GridRenderer
from orbitalengineer.ui.canvas.render.background import BackgroundRenderer
from orbitalengineer.ui.canvas.render.ellipse import EllipseRenderer
from orbitalengineer.ui.canvas.render.history import HistoryRenderer
from orbitalengineer.ui.canvas.render.particle import ParticleRenderer
from orbitalengineer.ui.canvas.render.pinpoint import PinpointRenderer
from orbitalengineer.ui.canvas.render.reticle import ReticleRenderer

from orbitalengineer.ipc.client import ClientSocketConnection
from orbitalengineer.ipc.clock import SimClock

HOVER_MARGIN = 15
DRAG_MARGIN = 50

class MoveParticleController:

    def __init__(self, canvas, camera, orbital:ClientSocketConnection, view_model:main.AppModel):
        self.canvas = canvas
        self.camera = camera
        self.orbital = orbital
        self.app = view_model
            
        self._prev_dx = 0
        self._prev_dy = 0

        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.set_exclusive(True) # ignore normal touch sequences
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        canvas.add_controller(drag)

    def _get_selection_box(self):
        x_start = min(self.app.drag_start[0], self.app.drag_end[0])
        x_end = max(self.app.drag_start[0], self.app.drag_end[0])
        y_start = min(self.app.drag_start[1], self.app.drag_end[1])
        y_end = max(self.app.drag_start[1], self.app.drag_end[1])
        return (x_start, x_end, y_start, y_end)

    def _update_selection(self):
        x_start, x_end, y_start, y_end = self._get_selection_box()
        self.app.selected_particles = np.where(
             (self.orbital.position.real <= x_end)
            &(self.orbital.position.real >  x_start)
            &(self.orbital.position.imag <= y_end)
            &(self.orbital.position.imag >  y_start)
            &((self.orbital.flags & flags.REMOVED) != flags.REMOVED)
        )[0].tolist()

    def on_drag_begin(self, gesture, start_x, start_y):
        event = gesture.get_last_event(None)
        if event is None: return

        device = event.get_device()
        if device.get_source() == Gdk.InputSource.TOUCHSCREEN:
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return

        x, y = self.camera.screen_to_world(start_x, start_y, self.canvas.get_width(), self.canvas.get_height())
        self.app.drag_start = (x, y)

        bodies = self.orbital.find_bodies_at(x, y, margin=5/self.camera.zoom)
        if len(bodies) == 0:
            self.app.selected_particles = None
            return
        
        self.app.dragging_particle = bodies[0]

    def on_drag_update(self, gesture, dx, dy):
        if not self.app.drag_start: return

        offset_x = (dx - self._prev_dx) / self.camera.zoom
        offset_y = (dy - self._prev_dy) / self.camera.zoom

        if offset_x > DRAG_MARGIN or offset_y > DRAG_MARGIN:
            self.app.engine.paused = True
        
        self.app.drag_end = (self.app.drag_start[0]+offset_x, self.app.drag_start[1]+offset_y)
        
        if self.app.dragging_particle is None:
            self._update_selection()
        else:
            self._prev_dx = dx
            self._prev_dy = dy
            offset = complex(offset_x, offset_y)
            self.orbital.rel_move(self.app.selected_particles, offset)
        
    def on_drag_end(self, gesture, start_x, start_y):
        self.app.dragging_particle = None
        self.app.drag_start = None
        self.app.drag_end = None
        self._prev_dx = 0
        self._prev_dy = 0

class MouseController:
    
    def __init__(self, canvas, camera, orbital, view_model:main.AppModel):
        self.canvas = canvas
        self.camera = camera
        self.orbital = orbital
        self.app = view_model        
        
        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self.on_motion)
        canvas.add_controller(motion)
 
    def on_motion(self, _ctrl, x, y):
        self.app.hover_position = (x, y)
        x, y = self.camera.screen_to_world(x, y, self.app.width, self.app.height)
        bodies = self.orbital.find_bodies_at(
            x, y,
            margin=HOVER_MARGIN/self.camera.zoom
        )
        if len(bodies) == 0:
            self.app.hovered_over_particle = None
            return
        self.app.hovered_over_particle = self.orbital.get_particle(bodies[0])


class OrbitalCanvas(Gtk.DrawingArea):

    def __init__(self, camera:Camera2D, app: main.AppModel, orbital:ClientSocketConnection, clock:SimClock, synth:ToneSynthController):
        super().__init__()
        self.camera = camera
        self.app = app
        self.clock = clock
        self.synth = synth
        self.orbital = orbital
        self.camera_ctl = Camera2DController(self, self.camera, self.app)
        self.mouse_controller = MouseController(self, self.camera, self.orbital, self.app)
        self.move_particle_ctl = MoveParticleController(self, self.camera, self.orbital, self.app)
        
        self.hud_renderers = [
            BackgroundRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            GridRenderer(self.app, self.camera, self.orbital, self.clock, self.synth)
        ]
        
        self.scene_renderers = [
            #HistoryRenderer(self.view, self.camera, self.orbital, self.clock),
            #ForceVectorRenderer(self.view, self.camera, self.orbital, self.clock),
            #CGroupRenderer(self.view, self.camera, self.orbital, self.clock),
            EllipseRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            ParticleRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            SelectionRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            ReticleRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            #PinpointRenderer(self.view, self.camera, self.orbital, self.clock, self.synth),
            #CGroupConnectionRenderer(self.view, self.camera, self.orbital, self.clock, self.synth),
        ]
        
        self.hud_fg_renderers = [
            DebugInfoRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            FocusInfoRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            HudClockRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            GPUStatusRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            WarningRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
            OSDRenderer(self.app, self.camera, self.orbital, self.clock, self.synth),
        ]
        
        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", self.on_click)
        self.add_controller(click_controller)
        
        def on_resize(self, width, height):
            self.app.width = width
            self.app.height = height
        self.connect("resize", on_resize)

        def on_tick(widget, frame_clock):
            self.queue_draw()
            return True
        self.add_tick_callback(on_tick)

    def on_click(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float):
        if n_press < 2: return
        
        event = gesture.get_current_event()
        if event is None: return
        
        state = event.get_modifier_state()
        ctrl_held = state & Gdk.ModifierType.CONTROL_MASK
        
        x, y = self.camera.screen_to_world(x, y, self.get_width(), self.get_height())
        bodies = self.orbital.find_bodies_at(x, y, margin=HOVER_MARGIN/self.camera.zoom)
        self.app.secondary_body = bodies[0] if len(bodies) > 0 else None
    
    def zoom_in(self):
        self.camera.zoom_at(0, 0, 0, 0, 0.9)

    def zoom_out(self):
        self.camera.zoom_at(0, 0, 0, 0, 1/0.9)

    def do_snapshot(self, snapshot: Gtk.Snapshot):
        now = self.clock.time()
        self.last_draw_time = now

        frame_clock = self.get_frame_clock()
        if frame_clock:
            self.app.frame_clock = frame_clock
            frame_clock = cast(Gdk.FrameClock, frame_clock)
            self.app.fps = frame_clock.get_fps()
                        
        width = self.get_allocated_width()
        height = self.get_allocated_height()
        
        try:
            if self.app.secondary_body is not None and self.app.follow_tracked_body:
                f = self.orbital.get_particle(self.app.secondary_body)
                fpos = f.get_position()
                self.camera.offset = [fpos.real, fpos.imag]

            cr = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))
            
            for r in self.hud_renderers:
                cr.save()
                try:
                    r.draw(cr, width, height)
                except Exception as e:
                    self.scene_renderers.remove(r)
                    raise e
                cr.restore()
            
            cr.save()
            cr.transform(self.camera.get_matrix(width, height))        

            for r in self.scene_renderers:
                cr.save()
                try:
                    r.draw(cr, width, height)
                except Exception as e:
                    self.scene_renderers.remove(r)
                    raise e
                cr.restore()
            
            cr.restore()
            
            for r in self.hud_fg_renderers:
                cr.save()
                try:
                    r.draw(cr, width, height)
                except Exception as e:
                    self.hud_fg_renderers.remove(r)
                    raise e
                cr.restore()

        except Exception as e:
            #print("EXCEPTION", e.wit)
            #raise SystemExit
            raise e
