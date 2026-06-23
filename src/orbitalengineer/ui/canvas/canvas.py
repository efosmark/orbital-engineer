from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.orbitalcl import SimController_CL
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.ui import model
from orbitalengineer.ui.canvas.render.hud_clock import HudClockRenderer
from orbitalengineer.ui.canvas.render.selection import SelectionRenderer
from orbitalengineer.ui.gtk4 import Gtk, Gdk, Graphene, Gsk
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.canvas.pz import Camera2D, Camera2DController
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

import numpy as np
import pyopencl as cl

HOVER_MARGIN = 15

class MoveParticleController:

    def __init__(self, canvas, camera, orbital, view_model):
        self.canvas = canvas
        self.camera = camera
        self.orbital = orbital
        self.view = view_model
        
        self._orig_particle_positions = None

        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.set_exclusive(True) # ignore normal touch sequences
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        canvas.add_controller(drag)

    def _get_selection_box(self):
        x_start = min(self.view.drag_start[0], self.view.drag_end[0])
        x_end = max(self.view.drag_start[0], self.view.drag_end[0])
        y_start = min(self.view.drag_start[1], self.view.drag_end[1])
        y_end = max(self.view.drag_start[1], self.view.drag_end[1])
        return (x_start, x_end, y_start, y_end)

    def _update_selection(self):
        x_start, x_end, y_start, y_end = self._get_selection_box()
        self.view.selected_particles = np.where(
             (self.orbital.position.real <= x_end)
            &(self.orbital.position.real >  x_start)
            &(self.orbital.position.imag <= y_end)
            &(self.orbital.position.imag >  y_start)
            &((self.orbital.flags & flags.REMOVED) != flags.REMOVED)
        )[0]

    def on_drag_begin(self, gesture, start_x, start_y):
        event = gesture.get_last_event(None)
        if event is None: return

        device = event.get_device()
        if device.get_source() == Gdk.InputSource.TOUCHSCREEN:
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return

        x, y = self.camera.screen_to_world(start_x, start_y, self.canvas.get_width(), self.canvas.get_height())
        self.view.drag_start = (x, y)

        bodies = self.orbital.find_bodies_at(x, y, margin=5/self.camera.zoom)
        if len(bodies) == 0:
            self.view.selected_particles = None
            return
        
        self.view.props.dragging_particle = bodies[0]

    def on_drag_update(self, gesture, dx, dy):
        if not self.view.drag_start: return

        offset_x = dx / self.camera.zoom
        offset_y = dy / self.camera.zoom
        
        if offset_x > 15 or offset_y > 15:
            self.view.props.paused = True
        
        self.view.drag_end = (self.view.drag_start[0]+offset_x, self.view.drag_start[1]+offset_y)
        
        self.view.props.paused = True
        if self.view.props.dragging_particle is None:
            self._update_selection()
        else:
            if self._orig_particle_positions is None:
                self._orig_particle_positions = self.orbital.position[self.view.selected_particles]
            offset = np.complex64(offset_x, offset_y)
            self.orbital.position[self.view.selected_particles] = self._orig_particle_positions + offset              
            cl.enqueue_copy(self.orbital.q, self.orbital.pos_cl, self.orbital.position)  
        
    def on_drag_end(self, gesture, start_x, start_y):
        self.view.props.dragging_particle = None
        self.view.props.dragging_particle_offset = (0, 0)
        self.view.props.drag_start = None
        self.view.props.drag_end = None
        self._orig_particle_positions = None

class MouseController:
    
    def __init__(self, canvas, camera, orbital, view_model):
        self.canvas = canvas
        self.camera = camera
        self.orbital = orbital
        self.view = view_model        
        
        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self.on_motion)
        canvas.add_controller(motion)

 
    def on_motion(self, _ctrl, x, y):
        self.view.hover_position = (x, y)
        x, y = self.camera.screen_to_world(x, y, self.view.width, self.view.height)
        bodies = self.orbital.find_bodies_at(
            x, y,
            margin=HOVER_MARGIN/self.camera.zoom
        )
        if len(bodies) == 0:
            self.view.hovered_over_particle = None
            return
        self.view.hovered_over_particle = self.orbital.get_particle(bodies[0])


class OrbitalCanvas(Gtk.DrawingArea):
    hud_renderers:list[renderer.Renderer]

    def __init__(self, camera:Camera2D, view: model.ViewModel, orbital:SimController_CL, clock:SimClock):
        super().__init__()
        
        self.camera = camera
        
        self.view = view
        self.clock = clock
        
        self.orbital = orbital
        self.camera_ctl = Camera2DController(self, self.camera, self.view)
        self.mouse_controller = MouseController(self, self.camera, self.orbital, self.view)
        self.move_particle_ctl = MoveParticleController(self, self.camera, self.orbital, self.view)
        
        self.hud_renderers = [
            BackgroundRenderer(self.view, self.camera, self.orbital, self.clock),
            GridRenderer(self.view, self.camera, self.orbital, self.clock),
        ]
        
        self.scene_renderers = [
            HistoryRenderer(self.view, self.camera, self.orbital, self.clock),
            ForceVectorRenderer(self.view, self.camera, self.orbital, self.clock),
            EllipseRenderer(self.view, self.camera, self.orbital, self.clock),
            ParticleRenderer(self.view, self.camera, self.orbital, self.clock),
            SelectionRenderer(self.view, self.camera, self.orbital, self.clock),
            ReticleRenderer(self.view, self.camera, self.orbital, self.clock),
            PinpointRenderer(self.view, self.camera, self.orbital, self.clock),
        ]
        
        self.hud_fg_renderers = [
            DebugInfoRenderer(self.view, self.camera, self.orbital, self.clock),
            FocusInfoRenderer(self.view, self.camera, self.orbital, self.clock),
            HudClockRenderer(self.view, self.camera, self.orbital, self.clock),
        ]
        
        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", self.on_click)
        self.add_controller(click_controller)
        
        def on_resize(self, width, height):
            self.view.width = width
            self.view.height = height
        self.connect("resize", on_resize)

        def on_tick(widget, frame_clock):
            self.queue_draw()
            return True
        self.add_tick_callback(on_tick)

    def on_click(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float):
        if n_press < 2:
            return
        
        event = gesture.get_current_event()
        if event is None:
            return
        state = event.get_modifier_state()
        ctrl_held = state & Gdk.ModifierType.CONTROL_MASK
        
        x, y = self.camera.screen_to_world(x, y, self.get_width(), self.get_height())
        bodies = self.orbital.find_bodies_at(x, y, margin=HOVER_MARGIN/self.camera.zoom)
        self.view.props.secondary_body = bodies[0] if len(bodies) > 0 else None
    
    def zoom_in(self):
        self.camera.zoom_at(0, 0, 0, 0, 0.9)

    def zoom_out(self):
        self.camera.zoom_at(0, 0, 0, 0, 1/0.9)

    def do_snapshot(self, snapshot: Gtk.Snapshot):
        now = self.clock.time()
        self.last_draw_time = now

        fps = self.get_frame_clock()
        if fps:
            self.view.fps = fps
        
        width = self.get_allocated_width()
        height = self.get_allocated_height()
        
        try:
            if self.view.secondary_body is not None and self.view.follow_tracked_body:
                f = self.orbital.get_particle(self.view.secondary_body)
                fpos = f.get_position()
                self.camera.offset = [fpos.real, fpos.imag]

            cr = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))
            
            for r in self.hud_renderers:
                cr.save()
                r.draw(cr, width, height)
                cr.restore()
            
            cr.save()
            cr.transform(self.camera.get_matrix(width, height))        

            for r in self.scene_renderers:
                cr.save()
                r.draw(cr, width, height)
                cr.restore()
            
            cr.restore()
            
            for r in self.hud_fg_renderers:
                cr.save()
                r.draw(cr, width, height)
                cr.restore()
        except Exception as e:
            #print("EXCEPTION", e.wit)
            #raise SystemExit
            raise e
