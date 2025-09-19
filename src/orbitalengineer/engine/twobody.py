import math
import numpy as np
from typing import Tuple, Optional, Dict

from orbitalengineer.engine import vec

def orbital_energy(mu, r, v):
    """Energy per unit mass of the orbiting body.

    Args:
        mu (float): Standard gravitational parameter
        r (float): Current distance from central body
        v (float): Relative velocity
    """
    return (v**2 / 2) - (mu / r)
    
def relative_state(r1: vec.Vec2, v1: vec.Vec2, r2: vec.Vec2, v2: vec.Vec2) -> Tuple[vec.Vec2, vec.Vec2]:
    """
    Relative state of body 2 with respect to body 1.
    r = r2 - r1, v = v2 - v1
    """
    r = vec.v_sub(r2, r1)
    v = vec.v_sub(v2, v1)
    return r, v

def standard_grav_param(M1: float, M2: float, G: float = 1.0) -> float:
    """
    Standard gravitational parameter μ = G (M1 + M2).
    Use consistent units for G, masses, positions, and velocities.
    """
    return G * (M1 + M2)

def universal_gravitation(G:float, M1:float, M2:float, r:float):
    """Newton's Law of Universal Gravitation

    Args:
        G (float): Gravitational constant
        M1 (float): Mass of first body
        M2 (float): Mass of second body
        r (float): Distance between the two bodies

    Returns:
        float: _description_
    """
    return G * ((M1 * M2) / r**2)

def grav_accel(G, M, r):
    return (G * M) / ((r**2) + 0.00001)

def orbital_velocity(G, M, r):
    return math.sqrt((G * M) / r)

def vis_viva(G, M, r, a):
    """Find the velocity of a bound object at a specific radius from a central mass.

    Latin for "Living force" — an early term for kinetic energy.
    Originator: Gottfried Leibniz

    Args: 
        G (float): Gravitational constant
        M (float): Mass of central body
        r (float): Current distance from central body
        a (float): Semi-major axis
    """
    return math.sqrt((G * M) * ((2.0/r) - (1/a)))

def vis_viva_velocity_vector(x:float, y:float, r:float, mu:float, a:float, prograde=True):
    v_mag = math.sqrt(mu * (2/r - 1/a))
    
    # Tangent unit vector: rotate (x, y) 90° counterclockwise (prograde)
    if prograde:
        t_x = -y / r
        t_y =  x / r
    else:
        t_x =  y / r
        t_y = -x / r

    # Velocity components
    vx = v_mag * t_x
    vy = v_mag * t_y
    
    return vx, vy


### AXES ###


def semi_major_axis(r: vec.Vec2, v: vec.Vec2, mu: float) -> float:
    """
    Semi-major axis from vis-viva:
      a = 1 / (2/|r| - |v|^2 / μ)
    Works for all conics (a>0 ellipse, a<0 hyperbola, a→∞ parabola).
    """
    rmag = vec.v_norm(r) + 1e-10
    v2 = vec.v_dot(v, v)
    return 1.0 / (2.0/rmag - v2/mu)

def semi_major_axis_2(mu, oe):
    """Size of the orbit (half of major axis)"""
    return - (mu / (2 * oe))

def ellipse_axes(a: float, e: float) -> Tuple[float, float]:
    """
    Major/minor axes for an ellipse.
      b = a * sqrt(1 - e^2)
    For near-circular e≈0, b≈a.
    """
    b2 = max(0.0, 1.0 - e*e)
    return a, a*math.sqrt(b2)

def periapsis_angle(e_unit: vec.Vec2) -> float:
    """
    Orientation angle of the major axis (radians).
    θ points toward periapsis.
    """
    return math.atan2(e_unit[1], e_unit[0])


### ECCENTRICITY ###


def specific_angular_momentum(x, y, vx, vy):
    """Specific angular momentum (scalar in 2D) in m**2/s"""
    return (x * vy) - (y * vx)

def ecc_vector(r: vec.Vec2, v: vec.Vec2, mu: float) -> vec.Vec2:
    """
    Eccentricity vector (points to periapsis). In 2D:
      e = ( (|v|^2 - μ/|r|) r - (r·v) v ) / μ
    """
    rmag = vec.v_norm(r)
    v2 = vec.v_dot(v, v)
    rv = vec.v_dot(r, v)
    ex = ((v2 - mu/rmag)*r[0] - rv*v[0]) / mu
    ey = ((v2 - mu/rmag)*r[1] - rv*v[1]) / mu
    return (ex, ey)

def eccentricity_vector(vx, vy, h, mu, r, x, y):
    """Eccentricity vector"""
    h = specific_angular_momentum(x, y, vx, vy)
    ex = (vy * h)/mu - (x/r)
    ey = -(vx * h)/mu - (y/r)
    return ex, ey

def ecc_scalar(e_vec: vec.Vec2) -> float:
    """Scalar eccentricity e = |e_vec|."""
    return vec.v_norm(e_vec)

def periapsis_direction(e_vec: vec.Vec2, eps: float = 1e-12) -> Optional[vec.Vec2]:
    """
    Unit vector e_unit pointing toward periapsis.
    Returns None if the orbit is (numerically) circular (e ≈ 0).
    """
    return vec.v_unit(e_vec, eps=eps)


def orbital_period(a:np.float64, mu:np.float64):
    r""" Time to complete one orbit (for bound orbits).
        $$ T = 2\pi \sqrt{\frac{a^3}{\mu}} $$
    """
    return (2.0 * np.pi) * np.sqrt( a**3 / mu )
    


class Elements2D:
    r"""
	Semi-major axis
        Size of the orbit (half of major axis).
    
    Eccentricity
        Shape of the orbit (0 = circle, < 1 = ellipse).
    
    Argument of periapsis
        Angle from reference direction to periapsis.
    
    True anomaly
        Current angular position from periapsis.
    
    Mean anomaly
        Linearized angle representing time since periapsis.
    
    Orbital period
        Time to complete one orbit (for bound orbits).
        $ T = 2\pi \sqrt{\frac{a^3}{\mu}} $
    
    Inclination
        [3D only] Tilt of orbit relative to reference plane.
    
    Longitude of ascending node
        [3D only] Angle from x-axis to node where orbit crosses reference plane upward.
    
    Specific angular momentum
        Magnitude of angular momentum per unit mass.
    
    Standard gravitational parameter
        G(M + m), but often approximated as if m=0.
    
    Specific orbital energy
        Energy per unit mass of the orbiting body.
    """
    def __init__(self, r1:np.complex128, r2:np.complex128, v1:np.complex128, v2:np.complex128, m1:np.float64, m2:np.float64):
        self.r1 = r1
        self.r2 = r2
        self.v1 = v1
        self.v2 = v2
        self.m1 = m1
        self.m2 = m2