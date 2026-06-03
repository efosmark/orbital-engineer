import math
import cairo
from orbitalengineer.ui.canvas import renderer

DEBUG_ELLIPSE = True
DEBUG_ELLIPSE_COLOR_RGBA = (0.3, 0.3, 0.5, 0.3)
DEBUG_ELLIPSE_COLOR_RGBA_2 = (0.5, 0.3, 0.3, 0.3)
DEFAULT_ELLIPSE_COLOR_RGBA = (0.2, 1.0, 0.6, 0.3)
DEFAULT_ELLIPSE_COLOR_FADED_RGBA = (0.2, 0.6, 0.6, 0.2)
SEMI_MAJOR_AXIS_RGBA = (0.5, 0.5, 0.5, 0.3)

# def ellipse_center_focus_primary(
#     r1: vec.Vec2, v1: vec.Vec2, r2: vec.Vec2, v2: vec.Vec2, M1: float, M2: float, G: float = 1.0,
#     circular_eps: float = 1e-12
# ) -> Ellipse:
#     """
#     Center of the secondary's ellipse when drawn with the PRIMARY at the focus.

#     Geometry:
#       focus = r1
#       center = focus - (a e) ê
#       where a from vis-viva, e and ê from the eccentricity vector.

#     Returns dict with:
#       cx, cy   : ellipse center
#       a, b     : semi-major/minor
#       e        : eccentricity
#       theta    : rotation angle (radians), major axis pointing to periapsis
#       mu       : standard gravitational parameter
#     """
#     r_rel, v_rel = twobody.relative_state(r1, v1, r2, v2)
#     mu = twobody.standard_grav_param(M1, M2, G)
#     e_vec = twobody.ecc_vector(r_rel, v_rel, mu)
#     e = twobody.ecc_scalar(e_vec)
#     a, b = twobody.ellipse_axes(r_rel, v_rel, mu, e)
    
#     # circular orbit case: center == focus, orientation arbitrary
#     if e < circular_eps:
#         b = a
#         print("CIRCULAR")
#         return Ellipse(cx=r1[0], cy=r1[1], a=a, b=b, e=0.0, theta=0.0, e_vec=e_vec, mu=mu, r_rel=r_rel)

#     cx, cy = twobody.ellipse_center(r1, e_vec, e, a)

#     e_unit = vec.unit(e_vec)
#     if e_unit is None:raise ValueError(f"{e_unit=}")
        
#     theta = twobody.argument_of_periapsis(e_unit)
#     return Ellipse(cx=cx, cy=cy, a=a, b=b, e=e, theta=theta, e_vec=e_vec, mu=mu, r_rel=r_rel)

def draw_ellipse(cr: cairo.Context, cx, cy, a, b, scale, *, angle:float|None=None, show_semimajor_axis=True):
    cr.save()
    cr.translate(cx, cy)
 
    cr.save()
    cr.new_path()
    cr.scale(a, b)
    cr.arc(0, 0, 1, 0, math.pi + math.pi*2.0)
    cr.restore()
    cr.stroke()

    if DEBUG_ELLIPSE:
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
        
        x,y = secondary.get_xy()
        cr.save()
        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.select_font_face("monospace")
        cr.set_font_size(10/self.camera.zoom)
        for i,line in enumerate(str(o).split(';')):
            cr.move_to(x + (10/self.camera.zoom), y + ((10/self.camera.zoom) * i))
            cr.show_text(line)
            cr.stroke()
        cr.restore()
        
        cr.save()
        cr.translate(*o.ellipse_center)
        cr.rotate(o.argument_of_periapsis)
        cr.set_line_width(1/self.camera.zoom)
        
        draw_ellipse(cr, 0, 0, o.axis_a, o.axis_b, self.camera.zoom, angle=o.argument_of_periapsis, show_semimajor_axis=True)
        cr.restore()

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_orbital_ellipse: return
        if self.view.secondary_body is not None:
            self.draw_ellipse_for_body(cr, self.view.secondary_body)
        b = self.view.hovered_over_particle
        if b is None or b.idx == self.view.secondary_body:
            return
        self.draw_ellipse_for_body(cr, b.idx, True)
