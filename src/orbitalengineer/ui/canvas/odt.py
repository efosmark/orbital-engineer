from contextlib import contextmanager
import math
from typing import Sequence
import cairo
from orbitalengineer.twobody import twobody
from orbitalengineer.ui.color import hex_to_rgba

DEBUG_ELLIPSE_COLOR_RGBA = (0.3, 0.3, 0.5, 0.9)
DEBUG_ELLIPSE_COLOR_RGBA_2 = (0.5, 0.3, 0.3, 0.3)
DEFAULT_ELLIPSE_COLOR_RGBA = (0.2, 1.0, 0.6, 0.3)
DEFAULT_ELLIPSE_COLOR_FADED_RGBA = (0.2, 0.6, 0.6, 0.1)
SEMI_MAJOR_AXIS_RGBA = (0.5, 0.5, 0.5, 0.3)
SEMI_MINOR_AXIS_RGBA = (0.5, 0.5, 0.5, 0.3)
APSIS_RGBA = (0.6, 0.6, 0.6, 0.2)

# hex_to_rgba("#06FFDE44") 

DEBUG_TRUE_ANOMALY_COLOR = hex_to_rgba("#418AFF44")
DEBUG_MEAN_ANOMALY_COLOR = hex_to_rgba("#FFD6335A")
DEBUG_ECCENTRIC_ANOMALY_COLOR = hex_to_rgba("#FF060628")

class OrbitDrawingTool:
    orbit:twobody.TwoBody

    @contextmanager
    def use_context(self):
        self.cr.save()
        yield self.cr
        self.cr.restore()

    def draw_ellipse(self, semi_major_axis, semi_minor_axis, scale, color, angle=math.pi*2.0):
        with self.use_context() as cr:
            cr.new_path()
            cr.scale(semi_major_axis, semi_minor_axis)
            cr.arc(0, 0, 1, 0, angle)
        cr.set_source_rgba(*color)
        cr.stroke()

    def draw_semimajor_axis(self, dash:Sequence[float]|None=None, color=SEMI_MAJOR_AXIS_RGBA):
        with self.use_context() as cr:
            cr.move_to(-self.orbit.semi_major_axis, 0)
            cr.line_to(+self.orbit.semi_major_axis, 0)
            cr.set_source_rgba(*color)
            if dash:
                cr.set_dash(dash)
            cr.stroke()

    def draw_semiminor_axis(self, dash:Sequence[float]|None=None, color=SEMI_MINOR_AXIS_RGBA):
        with self.use_context() as cr:
            cr.move_to(0, -self.orbit.semi_minor_axis)
            cr.line_to(0,  self.orbit.semi_minor_axis)
            cr.set_source_rgba(*color)
            if dash:
                cr.set_dash(dash)
            cr.stroke()

    def _draw_apsis(self, radius:float, apoapsis:bool=False):
        x = (self.orbit.semi_major_axis * (-1 if apoapsis else 1))
        with self.use_context() as cr:
            cr.set_source_rgba(*APSIS_RGBA)
            cr.move_to(x, 0)
            cr.arc(x, 0, radius, 0, 2 * math.pi)
            cr.fill()

    def _draw_aux_circle(self, axis, scale, color=DEBUG_ELLIPSE_COLOR_RGBA):
        with self.use_context() as cr:
            cr.new_path()
            cr.scale(axis, axis)
            cr.arc(0, 0, 1, 0, 2 * math.pi)
        cr.set_line_width(1.0/scale)
        cr.set_dash([2.0/scale, 4.0/scale])
        cr.set_source_rgba(*color)
        cr.stroke()

    def _draw_anomaly(self, anomaly, offset, color, scale):
        self.draw_ellipse(
            self.orbit.semi_major_axis - offset,
            self.orbit.semi_minor_axis - offset,
            scale,
            color,
            angle=anomaly
        )

    def _draw_anomaly_visualization(self, scale):
        with self.use_context() as cr:
            anomalys = [
                (self.orbit.true_anomaly, DEBUG_TRUE_ANOMALY_COLOR),
                (self.orbit.mean_anomaly, DEBUG_MEAN_ANOMALY_COLOR),
                (self.orbit.eccentric_anomaly, DEBUG_ECCENTRIC_ANOMALY_COLOR)
            ]
            
            anomaly_width = 3
            cr.set_line_width(anomaly_width/scale)
            for i, (anomaly, color) in enumerate(anomalys):
                offset = (anomaly_width * (i+1)) / scale
                self._draw_anomaly(anomaly, offset, color, scale)
                
    def draw_main_orbit_ellipse(self, scale):
        color = DEFAULT_ELLIPSE_COLOR_RGBA
        if self.orbit.falling_in:
            color = [color[1], color[0], color[2], color[3]]
        self.cr.set_line_width(1.0/scale)
        self.draw_ellipse(self.orbit.semi_major_axis, self.orbit.semi_minor_axis, scale, color)

    def draw_orbit(
        self,
        cr: cairo.Context,
        scale :float,                  # Size scale (for zoom factor)
        o: twobody.TwoBody,            # TwoBody details describing the orbit
        *,                             
        apsis_radius:float|None=None,  
        show_semimajor_axis=True,      
        show_anomaly:bool=False        
    ):
        self.cr = cr
        self.orbit = o
        cr.save()
        cr.translate(*o.ellipse_center)
        cr.rotate(o.argument_of_periapsis)

        with self.use_context() as cr:

            self.draw_main_orbit_ellipse(scale)

            if apsis_radius is not None:
                self._draw_apsis(apsis_radius, True)
                self._draw_apsis(apsis_radius, False)

            if show_anomaly:
                self._draw_anomaly_visualization(scale)

            self._draw_aux_circle(o.semi_major_axis, scale)
            self._draw_aux_circle(o.semi_minor_axis, scale)

            if show_semimajor_axis:
                self.draw_semimajor_axis(dash=[2.0/scale, 4.0/scale])
                self.draw_semiminor_axis(dash=[2.0/scale, 4.0/scale])

        cr.restore()