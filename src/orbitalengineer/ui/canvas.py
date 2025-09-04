from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import random
import statistics
import time

import numpy as np
import cairo

from orbitalengineer.engine.memory import HISTORY_DEPTH, BodyProxy, offset
from orbitalengineer.helpers import random_color
from orbitalengineer.ui.debug import create_debug_info
from orbitalengineer.ui.fmt import mag_format
from orbitalengineer.ui.grid import create_grid_surface
from orbitalengineer.ui.history import offset_index
from orbitalengineer.ui.panel import create_panel
from orbitalengineer.ui.ea4 import draw_ellipse
from orbitalengineer.ui.ell import ellipse_center_focus_primary
from orbitalengineer.ui.gtk4 import Gtk, Gdk, Graphene
from orbitalengineer.ui.pz import Camera2D, Camera2DController
from orbitalengineer.engine.orbitalnp import OrbitalSimController, OrbitalBody

## The minimum number of degrees betweeen each history point.
## Subsequent points below this threshold will be omitted.
## Setting this too high will cause history to be too short
MIN_HISTORY_DEG_CHANGE = 2.25

## Maximum history difference. If two history points are greater than 
## this, the history will be cleared.
MAX_HISTORY_DEG_CHANGE = 30.0

## The maximum number of points of history to track.
## High values can reduce performance.
MAX_HISTORY_POINTS = int(60 // MIN_HISTORY_DEG_CHANGE)

PANEL_PADDING = 8

@dataclass
class BodyMeta:
    id:int
    name:str
    color_fill:tuple[float,float,float]
    color_stroke:tuple[float,float,float] = (1, 1, 1)
    stroke_width:int = 1
    hist_rec:cairo.RecordingSurface|None = None
    hist_ctx:cairo.Context|None = None
    hist:list[tuple[float, float]] = field(default_factory=list)
    surface:cairo.RecordingSurface|None = None


class OrbitalCanvas(Gtk.DrawingArea):

    def __init__(self, orbital:OrbitalSimController):
        super().__init__()
        self.camera = Camera2D()
        self.controller = Camera2DController(self, self.camera)
        self.orbital = orbital
        self.body_meta:dict[int, BodyMeta] = {}
        self.frame_clock:Gdk.FrameClock|None = None
        self.step_id = 0
        self.tick_id = 0
        self.last_tick = -1
        
        # State fields
        self.secondary_idx:int|None = None
        self.primary_idx:int|None = None
        
        self.paused = False
        self.show_history = False
        self.show_map = True
        self.show_force_vectors = True
        self.show_orbital_ellipse = True
        self.show_reticle = True
        self.track_focused = True
        self.show_magnifier = True
        self.show_frame_clock = True
        self.show_focus_info = True
        
        self.t_render = deque(maxlen=10)
        self.avg_step_duration = 0
        self.avg_render_duration = 0
        
        # 
        self._grid_cache_key = None
        self._grid_surface = None
        
        self._texture = None
        
        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", self.on_click)
        self.add_controller(click_controller)
        
        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self.on_motion)
        self.add_controller(motion)
        self.mouse = (0, 0)

        draw_rate_hz = 20
        self.draw_interval = 1.0 / draw_rate_hz
        self.last_draw_time = 0.0

        self.add_tick_callback(self.on_tick)
        self._cached_surface = None

    def on_tick(self, widget, frame_clock):
        now = time.monotonic()
        if now - self.last_draw_time >= self.draw_interval:
            self.queue_draw()
        return True

    def on_click(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float):
        if n_press < 2:
            return
        
        event = gesture.get_current_event()
        if event is None:
            return
        state = event.get_modifier_state()
        ctrl_held = state & Gdk.ModifierType.CONTROL_MASK
        
        x, y = self.camera.screen_to_world(x, y, self.get_width(), self.get_height())
        bodies = self.orbital.find_bodies_at(x, y, margin=15/self.camera.zoom)
        if len(bodies) > 0:
            if ctrl_held:
                self.primary_idx = bodies[0]
            else:
                self.secondary_idx = bodies[0]
        else:
            if ctrl_held:
                self.primary_idx = None
            else:
                self.secondary_idx = None
    
    def on_motion(self, _ctrl, x, y):
        self.mouse = (x, y)
    
    def apply_accel(self, idx:int, ax:float, ay:float):
        self.orbital.inout_queue.append((idx, ax, ay))

    def zoom_in(self):
        self.camera.zoom_at(0, 0, 0, 0, 0.9)

    def zoom_out(self):
        self.camera.zoom_at(0, 0, 0, 0, 1/0.9)

    def add_element(self, el:OrbitalBody, color:tuple[float, float, float]=(1,1,1)):
        idx = self.orbital.add_body(el.x, el.y, el.mass, el.vx, el.vy, el.radius)
        color_stroke = random_color()
        self.body_meta[idx] = BodyMeta(idx, f"BDY-{hex(idx)}", color, color_stroke, stroke_width=int(random.uniform(1, 3)))
    
        m = self.body_meta[idx]
        w = h = int(el.radius*2)
        
        m.surface = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
        ctx = cairo.Context(m.surface)
        stroke_width = max(m.stroke_width, 1)
        radius = max(el.radius, 1/self.camera.zoom) + stroke_width
        ctx.set_source_rgb(*m.color_fill)
        ctx.arc(w/2, h/2, radius, 0, 2*math.pi)
        ctx.fill()
        
        ctx.set_source_rgb(*m.color_stroke)
        ctx.set_line_width(stroke_width)
        ctx.arc(w/2, h/2, radius, 0, 2*math.pi)
        ctx.stroke()

    def _draw_body_info(self, foc_idx, name:str):
        shm = self.orbital.shm
        
        b = self.orbital.body(foc_idx)
        x,y = b.get_xy(self.step_id)
        
        velocity = b.get_velocity(self.step_id)
        vx, vy = velocity.real, velocity.imag
        
        mag = np.abs(velocity)
        angle = math.degrees(math.atan2(vy, vx))
        
        disp = [
            ("name",     f"{self.body_meta[foc_idx].name} ({self.body_meta[foc_idx].id})"),
            ("(x, y)",    f"({mag_format(x)}, {mag_format(y)})"),
            ("mass",     f"{mag_format(shm.get_body_mass(self.step_id, foc_idx))}"),
            ("radius",   f"{mag_format(shm.get_body_radius(self.step_id, foc_idx))}"),
            ("(vx, vy)", f"({mag_format(vx)}/s, {mag_format(vy)}/s)"),
            ("|v|",      f"{mag_format(mag)}/s"),
            #("|a|",      f"{mag_format(np.abs(shm.a[foc_idx]))}/s²"),
            ("angle",    f"{angle:>.1f}°")
        ]
        lines = [f"{label:<8} {value:>18}" for label, value in disp]
        return create_panel(name, lines)

    def _draw_force_lines(self, cr:cairo.Context, i, forces, color=(1.0,1.0,1.0), N=10, scale=1.0):
        raise NotImplementedError()
        x1, y1 = self.orbital.shm.x[i], self.orbital.shm.y[i]
        
        if forces.size > N:
            fi = np.argpartition(forces, -N)[-N:]
        else:
            fi = np.arange(forces.size)
        
        fi = fi[(fi != self.secondary_idx)]
        fi = fi[self.orbital.shm.mass[fi] > 0]

        force_sum = np.sum(forces[fi])
        for fb in fi:
            x2, y2 = float(self.orbital.shm.x[fb]), float(self.orbital.shm.y[fb])
            fr = float(forces[fb]/force_sum)
            
            r,g,b = color[:3]
            cr.set_source_rgba(*[r, g, b, max(fr * 0.7, 0.05)])
            cr.set_line_width(max(fr, 1.0)/scale)
            cr.move_to(float(x1), float(y1))
            cr.line_to(x2, y2)
            cr.stroke()

    def _draw_reticle(self, cr:cairo.Context, idx, scale):
        body = self.orbital.body(idx)
        cr.set_line_cap(cairo.LineCap.ROUND)
        cr.set_line_width(2.5/scale)
        cr.set_source_rgb(0.7, 0.7, 0.7)
        ring_radius = body.get_radius(self.step_id) + (self.body_meta[body.idx].stroke_width/self.camera.zoom) + (10.0 / scale)
        
        pos = body.get_position(self.step_id)
        x, y = pos.real, pos.imag
        cr.arc(x, y, ring_radius, 0, 2*math.pi)
        cr.stroke()
        cr.save()
        try:
            cr.translate(x, y)
        except cairo.Error:
            print(f"ERROR. Cannot translate {x} and {y}")
            cr.restore()
            return
        reticle_count = 3
        for i in range(reticle_count):
            angle = (2.0*math.pi) / reticle_count
            
            cr.new_path()
            cr.move_to(0, ring_radius)
            cr.line_to(0, ring_radius + (ring_radius / 2))
            cr.stroke()
            cr.rotate(angle)
        
        cr.restore()
            
    def _draw_background(self, cr:cairo.Context, width, height):
        cr.set_source_rgb(0.005, 0.005, 0.02)
        cr.paint()

    def _draw_bodies(self, cr:cairo.Context, min_radius=1):
        for b in self.orbital:
            radius = b.get_radius(self.step_id)
            if radius == 0 or b.get_mass(self.step_id) == 0:
                continue
            
            pos = b.get_position(self.step_id)
            x,y = pos.real, pos.imag
            m = self.body_meta[b.idx]
                        
            radius = max(b.get_radius(self.step_id), 0.5/self.camera.zoom)
            
            cr.set_source_rgb(*m.color_fill)
            cr.arc(x, y, radius, 0, 2*math.pi)
            cr.fill()
            
            #cr.set_source_rgb(*m.color_stroke)
            #cr.set_line_width(1/self.camera.zoom)
            #cr.arc(x, y, radius, 0, 2*math.pi)
            #cr.stroke()
            
            if self.show_history:
                cr.set_line_width(2/self.camera.zoom)
                hp = self.orbital.shm.position
                hv = self.orbital.shm.velocity

                last_angle = None
                last_point = None

                steps = np.arange(max(0, int(self.tick_id-HISTORY_DEPTH+1)), self.tick_id+1) % HISTORY_DEPTH
                
                for tid, (v1,p1) in enumerate(zip(hv[steps, b.idx], hp[steps, b.idx])):
                    if p1 == 0: continue

                    angle = np.degrees(np.atan2(v1.imag, v1.real))
                    if last_angle is None or last_point is None:
                        last_angle = angle
                        last_point = p1
                        continue
                    
                    diff = abs(angle - last_angle)

                    if diff > 2.25:
                        alpha = float(0.2 + ((tid/min(int(HISTORY_DEPTH), self.tick_id)) * 0.4))
                        cr.set_source_rgba(*m.color_fill, alpha)
                        cr.move_to(last_point.real, last_point.imag)
                        cr.line_to(p1.real, p1.imag)
                        cr.stroke()
                        
                        last_angle = angle
                        last_point = p1

                if last_point is not None:
                    cr.move_to(last_point.real, last_point.imag)
                    cr.line_to(x, y)
                    cr.stroke()
                
    def _draw_scene(
        self,
        cr:cairo.Context,
        scale,
        show_force_vectors,
        show_history,
        show_orbital_ellipse,
        show_reticle
        ):
        
        if show_orbital_ellipse:
          self.draw_ellipse(cr, scale)
    
        #if self.secondary_idx is not None and show_force_vectors:
        #    forces = compute_forces(self.orbital.shm.x.size, self.orbital.shm, self.secondary_idx)
        #    self._draw_force_lines(cr, self.secondary_idx, forces, color=(1.0, 1, 1), N=5, scale=scale)        
        
        if self.secondary_idx is not None and show_reticle:
            self._draw_reticle(cr, self.secondary_idx, scale)
        
        self._draw_bodies(cr, 2/scale)

    def _compute_average_ellipse(self, primary:BodyProxy, secondary:BodyProxy):
        if secondary.get_mass(self.step_id) > primary.get_mass(self.step_id):
            primary, secondary = secondary, primary
        
        prim_position = primary.get_position(self.step_id)
        sec_position = secondary.get_position(self.step_id)
        
        prim_velocity = primary.get_velocity(self.step_id)
        sec_velocity = secondary.get_velocity(self.step_id)
        
        
        r1 = prim_position.real, prim_position.imag
        v1 = prim_velocity.real, prim_velocity.imag

        r2 = sec_position.real, sec_position.imag
        v2 = sec_velocity.real, sec_velocity.imag
        
        M1 = primary.get_mass(self.step_id)
        M2 = secondary.get_mass(self.step_id)
        
        if not M1 or not M2:
            return
        
        result = ellipse_center_focus_primary(r1, v1, r2, v2, M1, M2)
        return result
        
        if not hasattr(self, "_ellipse_history"):
            self._ellipse_history = defaultdict(list)
        
        hist = self._ellipse_history[secondary.idx]
        
        # If our primary body changed, clear the history
        if len(hist) > 0 and hist[-1]["pidx"] != primary.idx:
            hist.clear()
        
        # Sometimes b turns out to be 0 or -0. This is usually transient, so we'll
        # re-apply the last history point.
        is_valid = abs(result["b"]) > 0
        if is_valid:
            hist.append({**result, "pidx": primary.idx})
        else:
            hist.clear()

        #elif len(hist) > 0:
        #    hist.append({**hist[-1]})
        
        # We do not want to store too large of an averaging window
        while len(hist) > 3:
            hist.pop(0)
        
        if len(hist) > 0:
            # Average out the historical values
            return dict([
                (k, statistics.fmean([r[k] for r in hist]))
                for k in ('cx', 'cy', 'a', 'b', 'theta')
            ])

    def draw_ellipse(self, cr:cairo.Context, scale):
        if self.secondary_idx is None or self.primary_idx is None:
            return

        secondary = self.orbital.body(self.secondary_idx)
        primary = self.orbital.body(self.primary_idx)
        
        result = self._compute_average_ellipse(primary, secondary)
        if result is None or not result['a'] or not result['b']:
            return
        
        cr.set_source_rgba(1, 1, 1, 0.3)
        
        # We are ready to draw the ellipse. 
        # To make rotation simpler, we're set the reference to cx,cy
        cr.save()
        cr.translate(result["cx"], result["cy"])
        cr.rotate(result["theta"])
        cr.set_line_width(2/scale)
        draw_ellipse(cr, 0, 0, result["a"], result["b"], show_semimajor_axis=True)
        cr.restore()

    def _draw_frame_clock(self, cr:cairo.Context):
        accum = self.orbital.accum
        body_count = np.count_nonzero(self.orbital.shm.mass[offset(self.step_id),])
        
        frame_clock = self.get_frame_clock()
        if frame_clock is None:
            return
        
        fps = frame_clock.get_fps()
        frame_no = frame_clock.get_frame_counter()

        surface = create_debug_info(accum,
            body_count,
            fps,
            frame_no,
            self.avg_step_duration,
            self.avg_render_duration,
            self.get_width(),
            self.get_height()
        )
        
        cr.set_source_surface(surface)
        cr.paint()

    def draw_magnifier(self, cr, idx, rx, ry, rwidth, rheight, scale=1.0):
        raise NotImplementedError()
        cr.set_source_rgb(0, 0, 0)
        cr.rectangle(rx, ry, rwidth, rheight)
        cr.fill()
        
        cr.set_source_rgb(1,1,1)
        cr.rectangle(rx, ry, rwidth, rheight)
        cr.stroke()
        
        cr.rectangle(rx, ry, rwidth, rheight)
        cr.clip()
        
        cr.save()
        
        b = self.orbital.body(idx)
        
        min_width = 3.0
        max_width = rwidth * 0.85
        diam = b.get_radius(self.step_id) * 2.0
        if diam > max_width:
            scale = (max_width / diam)
        elif diam < min_width:
            scale = (min_width / diam)
        
        cr.scale(scale, scale)
        
        # Move the body to the origin
        cr.translate(-b.x, -b.y)
        
        # Offset the magnifier window origin
        cr.translate(rx/scale, ry/scale)
        
        # Center within the magnifier window
        cr.translate((rwidth/2/scale), (rheight/2/scale))

        self._draw_scene(cr, scale, self.show_force_vectors, self.show_history, self.show_orbital_ellipse, False)
        
        cr.restore()

    def _draw_focus_info(self, cr:cairo.Context):
        x = PANEL_PADDING
        y = PANEL_PADDING
        
        if self.secondary_idx is not None:
            sfc = self._draw_body_info(self.secondary_idx, "focused")
            cr.set_source_surface(sfc, x, y)
            cr.paint()
            y += sfc.get_height() + PANEL_PADDING
        
        if self.primary_idx is not None:
            sfc = self._draw_body_info(self.primary_idx, "primary")
            cr.set_source_surface(sfc, x, y)
            cr.paint()
            y += sfc.get_height() + PANEL_PADDING
        
        if self.secondary_idx is not None and self.primary_idx is not None:
            secondary = self.orbital.body(self.secondary_idx)
            primary = self.orbital.body(self.primary_idx)
            primary_x, primary_y = primary.get_xy(self.step_id)
            secondary_x, secondary_y = secondary.get_xy(self.step_id)
            
            dx, dy = primary_x - secondary_x, primary_y - secondary_y
            
            p_velocity = primary.get_velocity(self.step_id)
            primary_vx, primary_vy = p_velocity.real, p_velocity.imag
            
            s_velocity = secondary.get_velocity(self.step_id)
            secondary_vx, secondary_vy = s_velocity.real, s_velocity.imag
            
            dvx = primary_vx - secondary_vx
            dvy = primary_vy - secondary_vy
            
            disp = [
                ("distance",      f"{mag_format(math.hypot(dx, dy))}"),
                ("", ""),
                ("(Δvx, Δvy)",    f"({mag_format(dvx)}/s {mag_format(dvy)}/s"),
                ("|Δv|",          f"{mag_format(math.hypot(dvy,dvx))}/s"),
                ("rel angle",     f"{math.degrees(math.atan2(dvy, dvx)):.1f}°")
            ]
            lines = [f"{label:<10} {value:>16}" for label, value in disp]
            sfc = create_panel("focused vs primary", lines)
            cr.set_source_surface(sfc, x, y)
            cr.paint()

    def _draw_grid(self, cr:cairo.Context, width, height):
        x1, y1 = self.camera.screen_to_world(0, 0, width, height)
        x2, y2 = self.camera.screen_to_world(width, height, width, height)
        
        cache_key = (x1, y1, width, height)
        if self._grid_surface is None or self._grid_cache_key != cache_key:
            self._grid_cache_key = cache_key
            self._grid_surface = create_grid_surface(x1, y1, x2, y2, width, height)
        cr.set_source(self._grid_surface)
        cr.paint()

    def do_snapshot(self, snapshot: Gtk.Snapshot):
        width = self.get_allocated_width()
        height = self.get_allocated_height()

        if self._cached_surface is None or \
           self._cached_surface.get_width() != width or \
           self._cached_surface.get_height() != height:
            self._cached_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)


        now = time.monotonic()
        if now - self.last_draw_time >= self.draw_interval and self.tick_id > self.last_tick:
            self.last_draw_time = now
            cr = cairo.Context(self._cached_surface)
            
            start = time.perf_counter()
            self._draw_background(cr, width, height)

            if self.secondary_idx is not None and self.track_focused:
                f = self.orbital.body(self.secondary_idx)
                fpos = f.get_position(self.step_id)
                self.camera.offset = [fpos.real, fpos.imag]
                #self.camera.offset = [f.x, f.y]

            if self.show_map:
                self._draw_grid(cr, width, height)

            cr.save()
            cr.transform(self.camera.get_matrix(width, height))  
            self._draw_scene(
                cr,
                self.camera.zoom,
                self.show_force_vectors,
                self.show_history,
                self.show_orbital_ellipse,
                self.show_reticle
            )
            cr.restore()
            
            if self.show_focus_info:
                self._draw_focus_info(cr)

            if self.show_frame_clock:
                self._draw_frame_clock(cr)

            #if self.show_magnifier and self.secondary_idx is not None:
            #   self.draw_magnifier(cr, self.secondary_idx, 10, 120, 100, 100, 1)
        
            elapsed = time.perf_counter() - start
            self.t_render.append(elapsed)
        
        # Push cached surface into snapshot every frame
        rect = Graphene.Rect()
        rect.init(0, 0, width, height)
        cr_out = snapshot.append_cairo(rect)
        cr_out.set_source_surface(self._cached_surface, 0, 0)
        cr_out.paint()