from functools import cache
import math
import numpy as np
from typing import Tuple, Optional, Dict

from orbitalengineer.ui import vec

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


##############################################################################
## ORBITAL ELEMENTS
##############################################################################

def orbital_energy(mu:float, r:float, v:float) -> float:
    """Energy per unit mass of the orbiting body.

    Args:
        mu (float): Standard gravitational parameter
        r (float): Current distance from central body
        v (float): Relative velocity
    """
    return (v**2 / 2) - (mu / r)
    
def relative_state(r1: vec.Vec2, v1: vec.Vec2, r2: vec.Vec2, v2: vec.Vec2) -> tuple[vec.Vec2, vec.Vec2]:
    """
    Relative state of body 2 with respect to body 1.
    r = r2 - r1, v = v2 - v1
    """
    r = vec.sub(r2, r1)
    v = vec.sub(v2, v1)
    return (float(r[0]), float(r[1])), (float(v[0]), float(v[1]))

def standard_grav_param(M1: float, M2: float, G: float = 1.0) -> float:
    """ Standard gravitational parameter (μ = G * (M1 + M2))"""
    return G * (M1 + M2)

def ellipse_axes(r: vec.Vec2, v: vec.Vec2, mu: float, e: float, eps: float = 1e-12) -> tuple[float, float]:
    """ Major/minor axes for an ellipse.

    Axes from vis-viva:
      a = 1 / (2/|r| - |v|^2 / mu)
      b = a * sqrt(1 - e^2)
    
    Args:
        r: Orbital state position vector
        v: Orbital state velocity vector
        mu: Standard gravitational parameter
        e: Eccentricity (scalar)
    """
    rmag = vec.norm(r) + eps
    v2 = vec.dot(v, v)
    a = 1.0 / (2.0/rmag - v2/mu)
    b = a*math.sqrt(max(0.0, 1.0 - e*e))
    return a, b

def argument_of_periapsis(e_unit: vec.Vec2) -> float:
    """ Orientation angle of the major axis (radians)."""
    return math.atan2(e_unit[1], e_unit[0])

def specific_angular_momentum(r_vec:vec.Vec2, v_vec:vec.Vec2) -> float:
    r"""
    $$ h = x v_y - y v_x $$
    """
    return (r_vec[0] * v_vec[1]) - (r_vec[1] * v_vec[0])

def true_anomaly(e_vec:vec.Vec2, r_vec:vec.Vec2) -> float|None:
    e_unit = vec.unit(e_vec)
    if e_unit is None: return
    p_angle = argument_of_periapsis(e_unit)
    theta = (math.atan2(r_vec[1], r_vec[0]) - p_angle) + (math.pi)
    if theta < 0:
        theta += math.pi * 2.0
    return theta % (math.pi * 2.0)

def eccentric_anomaly(true_anom:float, e_vec:vec.Vec2) -> float:
    r""" 
    $$\tan E = \frac{\sin E}{\cos E} = \frac{\sqrt{1 - e^2} \cdot \sin{f}}{e + \cos{f}}$$
    """
    e = vec.norm(e_vec)
    sin_E = math.sqrt(1 - e**2) * math.sin(true_anom)
    cos_E = e + math.cos(true_anom)
    E = math.atan2(sin_E, cos_E)
    if E < 0:
        E = (math.pi * 2.0) + E
    return E 

def mean_anomaly(e_vec:vec.Vec2, E:float) -> float:
    """Kepler's Equation"""
    e = vec.norm(e_vec)
    return E - (e * math.sin(E))

def ecc_vector(r: vec.Vec2, v: vec.Vec2, mu: float) -> vec.Vec2:
    """ Eccentricity vector
    
    Args:
        r: Orbital state position vector
        v: Orbital state velocity vector
        mu: Standard gravitational parameter
    """
    rmag = vec.norm(r)
    v2 = vec.dot(v, v)
    rv = vec.dot(r, v)
    ex = ((v2 - mu/rmag)*r[0] - rv*v[0]) / mu
    ey = ((v2 - mu/rmag)*r[1] - rv*v[1]) / mu
    return (ex, ey)

def ecc_scalar(e_vec: vec.Vec2) -> float:
    """Scalar eccentricity e = |e_vec|."""
    return vec.norm(e_vec)

def orbital_period(a:float, mu:float) -> float:
    r""" Time to complete one orbit (for bound orbits).
    
    $$ T = 2\pi \sqrt{\frac{a^3}{\mu}} $$
    """
    return (2.0 * np.pi) * np.sqrt( a**3 / mu )
    

def periapsis_direction(e_vec: vec.Vec2, eps: float = 1e-12) -> Optional[vec.Vec2]:
    """
    Unit vector e_unit pointing toward periapsis.
    Returns None if the orbit is (numerically) circular (e ≈ 0).
    """
    return vec.unit(e_vec, eps=eps)


def is_retrograde(r_vec:vec.Vec2, v_vec:vec.Vec2) -> bool:
    """ Determine the direction of the orbit.
    
    The position vector points 'outward' from the focus/body, 
    and velocity shows which way the object is moving around that point. The 
    sign of `r x v` tells you whether angular momentum points CW or CCW.
    
    Positive value = velocity is rotated clockwise from position.
    Negative value = velocity is rotated counterclockwise from position.
    """
    return specific_angular_momentum(r_vec, v_vec) < 0


def barycenter(r1: vec.Vec2, r2: vec.Vec2, M1: float, M2: float) -> vec.Vec2:
    """Barycenter (center of mass)."""
    inv = 1.0/(M1 + M2)
    return ((M1*r1[0] + M2*r2[0]) * inv, (M1*r1[1] + M2*r2[1]) * inv)

def ellipse_center(r1:vec.Vec2, e_vec:vec.Vec2, e:float, a:float) -> vec.Vec2:
    e_unit = vec.unit(e_vec)
    if e_unit is None:raise ValueError()
    cx = r1[0] - a*e*e_unit[0]
    cy = r1[1] - a*e*e_unit[1]
    return (float(cx), float(cy))

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
    
    Specific angular momentum
        Magnitude of angular momentum per unit mass.
    
    Standard gravitational parameter
        G(M + m), but often approximated as if m=0.
    
    Specific orbital energy
        Energy per unit mass of the orbiting body.
    
    Inclination
        [3D only] Tilt of orbit relative to reference plane.
    
    Longitude of ascending node
        [3D only] Angle from x-axis to node where orbit crosses reference plane upward.
    """
    
    def __init__(self, r1:np.complex128, r2:np.complex128, v1:np.complex128, v2:np.complex128, m1:np.float64, m2:np.float64):
        self.r1 = r1
        self.r2 = r2
        self.v1 = v1
        self.v2 = v2
        self.m1 = m1
        self.m2 = m2