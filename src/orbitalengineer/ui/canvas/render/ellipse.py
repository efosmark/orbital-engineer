import math
import cairo
from orbitalengineer.engine.orbitalnp.integrator.conf import GRAV_DEFAULT
from orbitalengineer.engine.orbitalnp.integrator.kick import accel
from orbitalengineer.engine.simcontroller import Particle
from orbitalengineer.ui.canvas import renderer


DEFAULT_ELLIPSE_COLOR_RGBA = (0.2, 1.0, 0.6, 0.3)
DEFAULT_ELLIPSE_COLOR_FADED_RGBA = (0.2, 0.6, 0.6, 0.2)
SEMI_MAJOR_AXIS_RGBA = (0.5, 0.5, 0.5, 0.3)


def draw_ellipse(cr, cx, cy, a, b, *, show_semimajor_axis=True):
    cr.new_path()
    cr.save()
    cr.translate(cx, cy)

    cr.scale(a, b)
    cr.arc(0, 0, 1, 0, 2 * math.pi)
    cr.restore()

    cr.stroke()
    
    if show_semimajor_axis:
        # Draw the major axis
        cr.move_to(cx-a,cy)
        cr.line_to(cx+a, cy)
        cr.set_source_rgba(SEMI_MAJOR_AXIS_RGBA)
        cr.set_dash([8.0, 4.0])
        cr.stroke()
        cr.set_dash([])


import math
from typing import Tuple, Optional, Dict

Vec2 = Tuple[float, float]

# ---------------------------
# basic 2D vector operations
# ---------------------------

def v_add(a: Vec2, b: Vec2) -> Vec2:
    """Elementwise a + b."""
    return (a[0] + b[0], a[1] + b[1])

def v_sub(a: Vec2, b: Vec2) -> Vec2:
    """Elementwise a - b (vector from b to a)."""
    return (a[0] - b[0], a[1] - b[1])

def v_dot(a: Vec2, b: Vec2) -> float:
    """Dot product a·b."""
    return a[0]*b[0] + a[1]*b[1]

def v_scale(a: Vec2, s: float) -> Vec2:
    """Scale vector a by scalar s."""
    return (a[0]*s, a[1]*s)

def v_norm(a: Vec2) -> float:
    """Euclidean norm |a|."""
    return math.hypot(a[0], a[1])

def v_unit(a: Vec2, eps: float = 1e-15) -> Optional[Vec2]:
    """
    Unit vector in the direction of a.
    Returns None if |a| is too small (to avoid divide-by-zero).
    """
    n = v_norm(a)
    if n < eps:
        return None
    return (a[0]/n, a[1]/n)


# ---------------------------
# two-body helpers
# ---------------------------

def relative_state(r1: Vec2, v1: Vec2, r2: Vec2, v2: Vec2) -> Tuple[Vec2, Vec2]:
    """
    Relative state of body 2 with respect to body 1.
    r = r2 - r1, v = v2 - v1
    """
    r = v_sub(r2, r1)
    v = v_sub(v2, v1)
    return r, v

def standard_grav_param(M1: float, M2: float, G: float = 1.0) -> float:
    """
    Standard gravitational parameter μ = G (M1 + M2).
    Use consistent units for G, masses, positions, and velocities.
    """
    return G * (M1 + M2)

def semi_major_axis(r: Vec2, v: Vec2, mu: float) -> float:
    """
    Semi-major axis from vis-viva:
      a = 1 / (2/|r| - |v|^2 / μ)
    Works for all conics (a>0 ellipse, a<0 hyperbola, a→∞ parabola).
    """
    rmag = v_norm(r) + 1e-10
    v2 = v_dot(v, v)
    return 1.0 / (2.0/rmag - v2/mu)

def ecc_vector(r: Vec2, v: Vec2, mu: float) -> Vec2:
    """
    Eccentricity vector (points to periapsis). In 2D:
      e = ( (|v|^2 - μ/|r|) r - (r·v) v ) / μ
    """
    rmag = v_norm(r) 
    v2 = v_dot(v, v)
    rv = v_dot(r, v)
    ex = ((v2 - mu/rmag)*r[0] - rv*v[0]) / mu
    ey = ((v2 - mu/rmag)*r[1] - rv*v[1]) / mu
    return (ex, ey)

def ecc_scalar(e_vec: Vec2) -> float:
    """Scalar eccentricity e = |e_vec|."""
    return v_norm(e_vec)

def periapsis_direction(e_vec: Vec2, eps: float = 1e-12) -> Optional[Vec2]:
    """
    Unit vector ê pointing toward periapsis.
    Returns None if the orbit is (numerically) circular (e ≈ 0).
    """
    return v_unit(e_vec, eps=eps)

def ellipse_axes(a: float, e: float) -> Tuple[float, float]:
    """
    Major/minor axes for an ellipse.
      b = a * sqrt(1 - e^2)
    For near-circular e≈0, b≈a.
    """
    b2 = max(0.0, 1.0 - e*e)
    return a, a*math.sqrt(b2)

def periapsis_angle(ê: Vec2) -> float:
    """
    Orientation angle of the major axis (radians).
    θ points toward periapsis.
    """
    return math.atan2(ê[1], ê[0])


# ---------------------------
# centers to draw ellipses
# ---------------------------

def ellipse_center_focus_primary(
    r1: Vec2, v1: Vec2, r2: Vec2, v2: Vec2, M1: float, M2: float, G: float = 1.0,
    circular_eps: float = 1e-12
) -> Dict[str, float]:
    """
    Center of the secondary's ellipse when drawn with the PRIMARY at the focus.

    Geometry:
      focus = r1
      center = focus - (a e) ê
      where a from vis-viva, e and ê from the eccentricity vector.

    Returns dict with:
      cx, cy   : ellipse center
      a, b     : semi-major/minor
      e        : eccentricity
      theta    : rotation angle (radians), major axis pointing to periapsis
    """
    r_rel, v_rel = relative_state(r1, v1, r2, v2)
    mu = standard_grav_param(M1, M2, G)
    a = semi_major_axis(r_rel, v_rel, mu)
    e_vec = ecc_vector(r_rel, v_rel, mu)
    e = ecc_scalar(e_vec)

    # circular orbit case: center == focus, orientation arbitrary
    if e < circular_eps:
        b = a
        return dict(cx=r1[0], cy=r1[1], a=a, b=b, e=0.0, theta=0.0)

    e_unit = periapsis_direction(e_vec)  # cannot be None if e>=eps
    if e_unit is None:
        raise ValueError(f"{e_unit=}")
    cx = r1[0] - a*e*e_unit[0]
    cy = r1[1] - a*e*e_unit[1]
    _, b = ellipse_axes(a, e)
    theta = periapsis_angle(e_unit)
    return dict(cx=cx, cy=cy, a=a, b=b, e=e, theta=theta)

def barycenter(r1: Vec2, r2: Vec2, M1: float, M2: float) -> Vec2:
    """Barycenter (center of mass)."""
    inv = 1.0/(M1 + M2)
    return ((M1*r1[0] + M2*r2[0]) * inv, (M1*r1[1] + M2*r2[1]) * inv)

def ellipse_center_about_barycenter_for_body(
    r1: Vec2, v1: Vec2, r2: Vec2, v2: Vec2, M1: float, M2: float, for_primary: bool,
    G: float = 1.0, circular_eps: float = 1e-12
) -> Dict[str, float]:
    """
    Center of a body's ellipse when both ellipses are drawn about the BARYCENTER.

    Shared geometry:
      ê from the system eccentricity vector (same ê for both bodies).
      R_com = barycenter(r1, r2)

    Individual semi-major axes about COM:
      a1 = a * M2/(M1+M2)  (primary's ellipse)
      a2 = a * M1/(M1+M2)  (secondary's ellipse)

    Centers:
      center1 = R_com - (a1 e) ê
      center2 = R_com - (a2 e) ê

    Returns dict: cx, cy, a, b, e, theta (for the chosen body).
    """
    r_rel, v_rel = relative_state(r1, v1, r2, v2)
    mu = standard_grav_param(M1, M2, G)
    a_sys = semi_major_axis(r_rel, v_rel, mu)
    e_vec = ecc_vector(r_rel, v_rel, mu)
    e = ecc_scalar(e_vec)

    Rcom = barycenter(r1, r2, M1, M2)

    if e < circular_eps:
        # circles centered at COM with radii a1 or a2; theta arbitrary
        if for_primary:
            a_body = a_sys * M2/(M1+M2)
        else:
            a_body = a_sys * M1/(M1+M2)
        return dict(cx=Rcom[0], cy=Rcom[1], a=a_body, b=a_body, e=0.0, theta=0.0)

    e_unit = periapsis_direction(e_vec)  # cannot be None if e>=eps
    if e_unit is None:
        raise ValueError(f"{e_unit=}")
    
    if for_primary:
        a_body = a_sys * M2/(M1+M2)
    else:
        a_body = a_sys * M1/(M1+M2)

    cx = Rcom[0] - a_body*e*e_unit[0]
    cy = Rcom[1] - a_body*e*e_unit[1]
    _, b = ellipse_axes(a_body, e)
    theta = periapsis_angle(e_unit)
    return dict(cx=cx, cy=cy, a=a_body, b=b, e=e, theta=theta)


class EllipseRenderer(renderer.Renderer):

    def identify_grav_influence(self, secondary:Particle):
        sb_pos = secondary.get_position()
        sb_mass = secondary.get_mass()
        strongest_body = None
        strongest_accel = 0
        for b in self.orbital:
            if b.idx == secondary.idx:
                continue
            b_mass = b.get_mass()
            if  b_mass < sb_mass:
                continue
            pos = b.get_position()
            acceleration = abs(accel(sb_mass, b_mass, (sb_pos - pos), GRAV_DEFAULT))
            if acceleration > strongest_accel:
                strongest_accel = acceleration
                strongest_body = b
        return strongest_body

    def draw_ellipse_for_body(self, cr:cairo.Context, body_id:int, faded:bool=False):
        secondary = self.orbital.get_particle(body_id)

        strongest_body = self.identify_grav_influence(secondary)
        if strongest_body is None:
            return
        
        primary = strongest_body
        
        result = self._compute_average_ellipse(primary, secondary)
        if result is None or not result['a'] or not result['b']:
            return
        
        scale = self.camera.zoom
        
        if faded:
            cr.set_source_rgba(*DEFAULT_ELLIPSE_COLOR_FADED_RGBA)
        else:
            cr.set_source_rgba(*DEFAULT_ELLIPSE_COLOR_RGBA)
            
        # We are ready to draw the ellipse. 
        # To make rotation simpler, we're set the reference to cx,cy
        cr.save()
        cr.translate(result["cx"], result["cy"])
        cr.rotate(result["theta"])
        cr.set_line_width(1/scale)
        draw_ellipse(cr, 0, 0, result["a"], result["b"], show_semimajor_axis=False)
        cr.restore()

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_orbital_ellipse:
            return

        if self.data.secondary_body is not None:
            self.draw_ellipse_for_body(cr, self.data.secondary_body)

        b = self.view.hovered_over_particle
        if b is None or b.idx == self.data.secondary_body:
            return

        self.draw_ellipse_for_body(cr, b.idx, True)

    def _compute_average_ellipse(self, primary:Particle, secondary:Particle):
        if secondary.get_mass() > primary.get_mass():
            primary, secondary = secondary, primary
        
        prim_position = primary.get_position()
        sec_position = secondary.get_position()
        
        prim_velocity = primary.get_velocity()
        sec_velocity = secondary.get_velocity()
        
        r1 = prim_position.real, prim_position.imag
        v1 = prim_velocity.real, prim_velocity.imag

        r2 = sec_position.real, sec_position.imag
        v2 = sec_velocity.real, sec_velocity.imag
        
        M1 = primary.get_mass()
        M2 = secondary.get_mass()
        
        if not M1 or not M2:
            return
        
        result = ellipse_center_focus_primary(r1, v1, r2, v2, M1, M2)
        return result
