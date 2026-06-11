import math
import cairo
from orbitalengineer.engine import twobody
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.color import hex_to_rgba

DEBUG_AUX_CIRCLES = False
DEBUG_ANOMALY_VALUES = False

DEBUG_ELLIPSE_COLOR_RGBA = (0.3, 0.3, 0.5, 0.3)
DEBUG_ELLIPSE_COLOR_RGBA_2 = (0.5, 0.3, 0.3, 0.3)
DEFAULT_ELLIPSE_COLOR_RGBA = (0.2, 1.0, 0.6, 0.3)
DEFAULT_ELLIPSE_COLOR_FADED_RGBA = (0.2, 0.6, 0.6, 0.18)
SEMI_MAJOR_AXIS_RGBA = (0.5, 0.5, 0.5, 0.3)
APSIS_RGBA = (0.6, 0.6, 0.6, 0.2)

# hex_to_rgba("#06FFDE44") 

DEBUG_TRUE_ANOMALY_COLOR = hex_to_rgba("#418AFF33")
DEBUG_MEAN_ANOMALY_COLOR = hex_to_rgba("#FFD6336A")
DEBUG_ECCENTRIC_ANOMALY_COLOR = hex_to_rgba("#FF060633")


def draw_ellipse(cr: cairo.Context, cx, cy, a, b, scale, o: twobody.TwoBody, *, apsis_radius:float|None=None, show_semimajor_axis=True):
    cr.save()
    cr.translate(cx, cy)
 
    if apsis_radius is not None:
        cr.save()
        cr.set_source_rgba(*APSIS_RGBA)
        for x in [cx+a, cx-a]:
            cr.move_to(x, cy)
            cr.arc(x, cy, apsis_radius, 0, 2 * math.pi)
            cr.fill()
        cr.restore()

    if DEBUG_ANOMALY_VALUES:

        anomalys = [
            (o.true_anomaly, DEBUG_TRUE_ANOMALY_COLOR),
            (o.mean_anomaly, DEBUG_MEAN_ANOMALY_COLOR),
            (o.eccentric_anomaly, DEBUG_ECCENTRIC_ANOMALY_COLOR)
        ]
        
        anomaly_width = 5
        cr.set_line_width(anomaly_width/scale)
        for i, (anomaly, color) in enumerate(anomalys):
            offset = (anomaly_width * (i+1)) / scale
            cr.save()
            cr.new_path()
            cr.scale(a + offset, b + offset)
            cr.arc(0, 0, 1, 0, anomaly)
            cr.restore()
            cr.set_source_rgba(*color)
            cr.stroke()
                
    cr.save()
    cr.new_path()
    cr.scale(a, b)
    cr.arc(0, 0, 1, 0, math.pi*2.0)
    cr.restore()
    cr.set_source_rgba(*DEFAULT_ELLIPSE_COLOR_RGBA)
    cr.set_line_width(2.0/scale)
    cr.stroke()

    if DEBUG_AUX_CIRCLES:
        cr.set_dash([2.0/scale, 4.0/scale])

        cr.set_source_rgba(*DEBUG_ELLIPSE_COLOR_RGBA)
        cr.save()
        cr.new_path()
        cr.scale(a, a)
        cr.arc(0, 0, 1, 0, 2 * math.pi)
        cr.restore()
        cr.stroke()

        cr.set_source_rgba(*DEBUG_ELLIPSE_COLOR_RGBA_2)
        cr.save()
        cr.new_path()
        cr.scale(b,b)
        cr.arc(0, 0, 1, 0, 2 * math.pi)
        cr.restore()
        cr.stroke()
    cr.restore()

    if show_semimajor_axis:
        # Draw the major axis
        cr.move_to(cx-a, cy)
        cr.line_to(cx+a, cy)
        cr.set_source_rgba(*SEMI_MAJOR_AXIS_RGBA)
        cr.set_dash([8.0, 4.0])
        cr.stroke()
        cr.set_dash([])


class EllipseRenderer(renderer.Renderer):

    def draw_ellipse_for_body(self, cr:cairo.Context, body_id:int, faded:bool=False):
        secondary = self.orbital.get_particle(body_id)
                
        if faded: cr.set_source_rgba(*DEFAULT_ELLIPSE_COLOR_FADED_RGBA)
        else:     cr.set_source_rgba(*DEFAULT_ELLIPSE_COLOR_RGBA)
        
        o = secondary.get_orbit_info()
        if not o: return
        
        cr.save()
        cr.translate(*o.ellipse_center)
        cr.rotate(o.argument_of_periapsis)
        cr.set_line_width(2/self.camera.zoom)
        draw_ellipse(cr, 0, 0,
            o.semi_major_axis,
            o.semi_minor_axis,
            self.camera.zoom,
            o,
            show_semimajor_axis=False
        )
        cr.restore()

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_orbital_ellipse: return
        if self.view.secondary_body is not None:
            self.draw_ellipse_for_body(cr, self.view.secondary_body)
        
        b = self.view.hovered_over_particle
        if b is not None and b.idx != self.view.secondary_body:
            self.draw_ellipse_for_body(cr, b.idx, True)
        
        # TESTING -- showing all secondary bodies orbiting focus
        #if self.view.secondary_body is not None:
        #    for b in self.orbital:
        #        if b.get_focus() == self.view.secondary_body:
        #            self.draw_ellipse_for_body(cr, b.idx, True)
