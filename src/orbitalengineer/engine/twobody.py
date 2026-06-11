from functools import cache
import math
import numpy as np
from numba import jit

TAU = math.pi * 2.0

Vec2 = tuple[float, float]
Point = tuple[float, float]

@jit(cache=False)
def vec_sub(a: Vec2, b: Vec2) -> Vec2:
    """Elementwise a - b (vector from b to a)."""
    return (a[0] - b[0], a[1] - b[1])

@jit(cache=False)
def vec_dot(a: Vec2, b: Vec2) -> float:
    """Dot product a·b."""
    return a[0]*b[0] + a[1]*b[1]

@jit(cache=False)
def vec_norm(a: Vec2) -> float:
    """Euclidean norm |a|."""
    return np.hypot(a[0], a[1])

@jit(cache=False)
def vec_unit(a: Vec2, eps: float = 1e-15) -> Vec2:
    """
    Unit vector in the direction of a.
    Returns None if |a| is too small (to avoid divide-by-zero).
    """
    n = vec_norm(a)
    if n < eps: return (0,0)
    return (a[0]/n, a[1]/n)


@jit(cache=False)
def universal_gravitation(G:float, M1:float, M2:float, r:float) -> float:
    """
    Newton's Law of Universal Gravitation
 
    Args:
        G (float): Gravitational constant
        M1 (float): Mass of first body
        M2 (float): Mass of second body
        r (float): Distance between the two bodies

    Returns:
        float: Gravitational force between two bodies
    """
    return G * ((M1 * M2) / r**2)

@jit(cache=False)
def vis_viva_velocity(x:float, y:float, r:float, mu:float, a:float, prograde=True) -> Vec2:
    """
    Find the velocity of a bound object at a specific radius from a central mass.

    Latin for "Living force" - an early term for kinetic energy.
    Originator: Gottfried Leibniz

    Args: 
        x (float): Primary body (focus) x-coordinate
        y (float): Primary body (focus) y-coordinate
        r (float): Current distance from central body
        mu (float): Standard graviatational parameter
        a (float): Semi-major axis
    
    Returns:
        tuple[float,float]: The requisite velocity of an object that will match the orbital characteristics.
    """
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

@jit(cache=False)
def relative_state(r1: Vec2, v1: Vec2, r2: Vec2, v2: Vec2) -> tuple[Vec2, Vec2]:
    """
    Orbital state vectors for position and velocity.
    
    Args:
        r1 (tuple[float,float]):  Position of body-1 (x,y)
        v1 (tuple[float,float]):  Velocity of body-1 (x,y)
        r2 (tuple[float,float]):  Position of body-2 (x,y)
        v2 (tuple[float,float]):  Velocity of body-2 (x,y)
    
    Returns:
        tuple[tuple[float,float], tuple[float,float]]: Relative state of body 2 with respect to body 1.
    """
    r = vec_sub(r2, r1)
    v = vec_sub(v2, v1)
    return (float(r[0]), float(r[1])), (float(v[0]), float(v[1]))

@jit(cache=False)
def orbital_energy(mu:float, r:float, v:float) -> float:
    """
    Energy per unit mass of the orbiting body.
    Specific orbital energy (ε) is the total mechanical energy per unit mass of an orbiting body.

    Interpreting the values:
    
        | Value | Orbit Type | Outcome            |
        | ----- | ---------- | -------------------|
        | ε < 0 | Elliptical | Bound orbit        |
        | ε = 0 | Parabolic  | Escape trajectory  |
        | ε > 0 | Hyperbolic | Unbound            |

    Args:
        mu (float): Standard gravitational parameter
        r (float): Current distance from central body
        v (float): Relative velocity
    
    Returns:
        float: The total mechanical energy per unit mass of an orbiting body.
    """ 
    return (v**2 / 2) - (mu / r)

@jit(cache=False)
def standard_grav_param(m1:float, m2:float, G:float=1.0) -> float:
    """
    Standard gravitational parameter (μ or mu).
    This is the sum of the masses for the two bodies, multiplied by G:

        μ = G * (M1 + M2)
    
    Args:
        m1 (float): Mass of the first body (kg)
        m2 (float): Mass of the second body (kg)
        G (float): Gravitational constant (default 1.0)
    
    Returns:
        float: Standard grav. parameter, in unit `m^3 ⋅ s^-2`
    """
    return G * (m1 + m2)

@jit(cache=False)
def ellipse_axes(r:Vec2, v:Vec2, mu:float, e:float, eps:float=1e-12) -> tuple[float, float]:
    """
    Semi-major/semi-minor axes for an ellipse.

    Axes from vis-viva:
    
        a = 1 / (2/|r| - |v|^2 / μ)
        b = a * sqrt(1 - e^2)
    
    Args:
        r (tuple[float,float]): Orbital state position vector
        v (tuple[float,float]): Orbital state velocity vector
        mu (float): Standard gravitational parameter
        e (float): Eccentricity (scalar)
    
    Returns:
        tuple[float, float]: The pair of both semi-major and semi-minor axes (a, b).
    """
    rmag = vec_norm(r) + eps
    v2 = vec_dot(v, v)
    a = 1.0 / (2.0/rmag - v2/mu)
    b = a*math.sqrt(max(0.0, 1.0 - e*e))
    return a, b

@jit(cache=False)
def eccentricity_vec(r: Vec2, v: Vec2, mu: float) -> Vec2:
    """
    Eccentricity vector that defines the shape of the orbit.
    
    To get a scalar value, get the norm of the vector. (0 = circle, < 1 = ellipse)
    
    Args:
        r (tuple[float,float]): Orbital state position vector
        v (tuple[float,float]): Orbital state velocity vector
        mu (float): Standard gravitational parameter
    
    Returns:
        tuple[float,float]: Eccentricity as an (x,y) vector
    """
    rmag = vec_norm(r)
    v2 = vec_dot(v, v)
    rv = vec_dot(r, v)
    ex = ((v2 - mu/rmag)*r[0] - rv*v[0]) / mu
    ey = ((v2 - mu/rmag)*r[1] - rv*v[1]) / mu
    return (ex, ey)

@jit(cache=False)
def argument_of_periapsis(e_unit: tuple[float, float]) -> float:
    """
    Orientation angle of the major axis from reference direction to periapsis.
    
    Args:
        e_unit (tuple[float, float]): The unit vector of the eccentricity
    
    Returns:
        float: Angle in radians of the major axis off of the periapsis.
    """
    return np.atan2(e_unit[1], e_unit[0])

@jit(cache=False)
def specific_angular_momentum(r_vec:Vec2, v_vec:Vec2) -> float:
    """
    Magnitude of angular momentum per unit mass.

    Args:
        r_vec (tuple[float,float]): Orbital state position
        v_vec (tuple[float,float]): Orbital state velocity

    Returns:
        float: Angular momentum of the body divided by its mass
    """
    return (r_vec[0] * v_vec[1]) - (r_vec[1] * v_vec[0])

@jit(cache=False)
def true_anomaly(r_vec:Vec2, arg_of_periapsis:float) -> float:
    """
    Current angular position with respect to periapsis, encoded in radians.

    Args:
        r_vec (tuple[float,float]): Relative state of body 2 with respect to body 1
        arg_of_periapsis (float): Angle between periapsis and major axis in radians
    
    Returns:
        float: True anomaly in radians
    """
    T_0 = (math.atan2(r_vec[1], r_vec[0]) - arg_of_periapsis)
    if T_0 < 0: T_0 += TAU
    return T_0 % TAU

@jit(cache=False)
def eccentric_anomaly(true_anom:float, e_vec:Vec2) -> float:
    r""" 
    Eccentric Anomaly (E) is the bridge between circular and elliptical motion. 
    
    $$\tan E = \frac{\sin E}{\cos E} = \frac{\sqrt{1 - e^2} \cdot \sin{f}}{e + \cos{f}}$$
    
    Args:
        true_anom (float): True anomaly in radins
        e_vec (tuple[float, float]): Eccentricity vector
    
    Returns:
        float: The eccentric anomaly value in radians.
    """
    e = vec_norm(e_vec)
    sin_E = math.sqrt(1 - e**2) * math.sin(true_anom)
    cos_E = e + math.cos(true_anom)
    E = math.atan2(sin_E, cos_E)
    if E < 0:
        E = TAU + E
    return E 

@jit(cache=False)
def mean_anomaly(e_vec:Vec2, E:float) -> float:
    """
    A time-based angular parameter that describes how far along the orbit the 
    object would be if it moved at a constant angular speed around a circle, 
    rather than a variable speed on an ellipse.
    
    This uses Kepler's Equation to solve for the mean anomaly.
    
    Args:
        e_vec (tuple[float,float]): Orbital eccentricity vector
        E (float): The eccentric anomaly value in radians
    
    Returns:
        float: Mean anomaly, measured in radians
    """
    e = vec_norm(e_vec)
    return E - (e * math.sin(E))


@jit(cache=False)
def orbital_period(a:float, mu:float) -> float:
    r"""
    Time to complete one orbit (for bound orbits).
    
        $$ T = 2\pi \sqrt{\frac{a^3}{\mu}} $$
    
    Args:
        a (float): Semi-major axis, measured in meters
        mu (float): Standard gravitational parameter
    
    Returns:
        float: Orbital period, measured in seconds
    """
    return (2.0 * np.pi) * np.sqrt( a**3 / mu )


@jit(cache=False)
def mean_motion(T:float) -> float:
    """ 
    Angular speed required for a body to complete one orbit.
    
    Aassumes constant speed in a circular orbit (see: auxiliary circle / eccentric anomaly)
    which completes in the same time as the variable speed, elliptical orbit of the actual body 
    
    Args:
        T (float): Orbital period, in seconds
    
    Returns:
        float: Mean motion angular speed
    """
    return TAU / T


@jit(cache=False)
def periapsis_direction(e_vec: Vec2, eps:float=1e-12) -> Vec2:
    """
    Unit vector e_unit pointing toward periapsis.
    Returns None if the orbit is (numerically) circular (e ≈ 0).
    
    Args:
        e_vec (tuple[float,float]): Orbital eccentricity vector
    
    Returns:
        float: Unit vector pointing toward periapsis.
    """
    return vec_unit(e_vec, eps=eps)


@jit(cache=False)
def is_retrograde(r_vec:Vec2, v_vec:Vec2) -> bool:
    """
    Determine the direction of the orbit.
    
    The position vector points 'outward' from the focus/body, 
    and velocity shows which way the object is moving around that point. The 
    sign of `r x v` tells you whether angular momentum points CW or CCW.
    
    Positive value = velocity is rotated clockwise from position.
    Negative value = velocity is rotated counterclockwise from position.
    
    Args:
        r_vec (tuple[float,float]): Relative position of body 2 with respect to body 1
        v_vec (tuple[float,float]): Relative velocity of body 2 with respect to body 1
    
    Returns:
        bool: `True` if the orbit is retrograde, `False` otherwise
    """
    return specific_angular_momentum(r_vec, v_vec) < 0


@jit(cache=False)
def barycenter(r1: Vec2, r2: Vec2, m1: float, m2: float) -> Vec2:
    """
    Barycenter (center of mass).
    
    Args:
        r1 (Point): Position (x,y) of the first body
        r2 (Point): Position (x,y) of the second body
        m1 (float): Mass of the first body
        m2 (float): Mass of the second body
    
    Returns:
        Point: The (x,y) center-of-mass.
    """
    inv = 1.0/(m1 + m2)
    return ((m1*r1[0] + m2*r2[0]) * inv, (m1*r1[1] + m2*r2[1]) * inv)


@jit(cache=False)
def ellipse_center(r1:Vec2, e_vec:Vec2, e:float, a:float) -> Vec2:
    """
    Center point of the orbital ellipse as an (x,y) vector.
    
    Args:
        r1 (Point): Postion of the primary body as (x,y)
        e_vec (Vec2): Orbital eccentricity vector
        e (float): Orbital eccentricity magnitude
        a (float): Semi-major axis
    
    Returns:
        Point: Pair of values for the (x,y) position
    """
    e_unit = vec_unit(e_vec) or (0,0)
    cx = r1[0] - a*e*e_unit[0]
    cy = r1[1] - a*e*e_unit[1]
    return cx, cy


@jit(cache=False)
def escape_velocity(r_rel: Vec2, grav_param: float) -> float:
    """
    Given the orbital state, find the velocity required to escape a bound orbit.
    Essentially, this finds the Δv from where the specific orbital energy (ε) is 0, and
    can work in reverse to find the velocity required to be bound (negative value).
    
    Args:
        r_rel (tuple[float,float]): Relative position of body 2 with respect to body 1
        mu (float): Standard gravitational parameter
    
    Returns:
        float: Velocity that would be required to escape a bound orbit
    """
    return np.sqrt((2 * grav_param) / vec_norm(r_rel))


cached_property = cache(property)

class TwoBody:
    r"""
    
    Body properties:
    
        r1 (Point):  Location (x,y) of the focus (primary) body
        r2 (Point):  Location (x,y) of the secondary (orbiting) body

        v1 (Vec2):   Velocity (x,y) of the focus (primary) body
        v2 (Vec2):   Velocity (x,y) of the secondary (orbiting) body
        
        m1 (float):  Mass of the first body
        m2 (float):  Mass of the second body
    
    Orbital state properties:
    
        r_rel
        v_rel
        standard_grav_param

    Ellipse shape properties:

        ellipse_center
        argument_of_periapsis
        eccentricity
        semi_major_axis
        semi_minor_axis
        apoapsis
        periapsis
    
    """
    
    accuracy:float = 0.0
    
    # Body properties
    r1: Point
    r2: Point
    v1: Vec2
    v2: Vec2
    m1: float
    m2: float
    
    # Orbital State
    r_rel: tuple[float, float]
    v_rel: tuple[float, float]

    # Ellipse shape
    ellipse_center: tuple[float, float]
    argument_of_periapsis: float
    eccentricity_vec: tuple[float, float]
    eccentricity: float
    semi_major_axis: float
    semi_minor_axis: float
        
    # Motion over time
    standard_grav_param: float
    orbital_period: float
    mean_motion: float

    orbital_energy: float
    true_anomaly: float
    mean_anomaly: float    
    
    distance: float
    eccentric_anomaly: float
    v_esc: float
    direction: str
    time_periapsis: float

    def __init__(self, r1: Vec2, r2: Vec2, v1: Vec2, v2: Vec2, m1: float, m2: float):
        self.r1, self.r2 = r1, r2
        self.v1, self.v2 = v1, v2
        self.m1, self.m2 = m1, m2
        
        self.r_rel, self.v_rel = relative_state(r1, v1, r2, v2)
        self.distance = vec_norm(self.r_rel)
        self.standard_grav_param = standard_grav_param(m1, m2, 1.0)
 
        self.orbital_energy = orbital_energy(self.standard_grav_param, vec_norm(self.r_rel), vec_norm(self.v_rel))        
        self.eccentricity_vec = eccentricity_vec(self.r_rel, self.v_rel, self.standard_grav_param)
        self.e_unit = vec_unit(self.eccentricity_vec) or (0,0)
        self.eccentricity = vec_norm(self.eccentricity_vec)
        self.is_bound = self.eccentricity < 1.0
        
        self.semi_major_axis, self.semi_minor_axis = ellipse_axes(self.r_rel, self.v_rel, self.standard_grav_param, self.eccentricity)
        self.ellipse_center = ellipse_center(r1, self.eccentricity_vec, self.eccentricity, self.semi_major_axis)
        self.argument_of_periapsis = argument_of_periapsis(self.e_unit)
        self.true_anomaly = true_anomaly(self.r_rel, self.argument_of_periapsis)
        self.eccentric_anomaly = eccentric_anomaly(self.true_anomaly, self.eccentricity_vec)
        self.mean_anomaly = mean_anomaly(self.eccentricity_vec, self.eccentric_anomaly)
        self.orbital_period = orbital_period(self.semi_major_axis, self.standard_grav_param)
        self.mean_motion = mean_motion(self.orbital_period)

        self.direction = "prograde"
        self.time_periapsis = (self.orbital_period * (self.mean_anomaly / (np.pi*2.0)))
        if is_retrograde(self.r_rel, self.v_rel):
            self.direction = "retrograde"
            self.time_periapsis = self.orbital_period - self.time_periapsis
        
        self.v_esc = escape_velocity(self.r_rel, self.standard_grav_param)

    def __str__(self):
        fields = [
            ('ε',  f"{self.orbital_energy:.1e}"),
            ('μ',  f"{self.standard_grav_param:.1e}"),
            ('θ₀', f"{self.true_anomaly:.2f}"),
            ('e',  f"{self.eccentricity:.2f}"),
            ('ω',  f"{self.argument_of_periapsis:.2f}"),
            ('M₀', f"{self.mean_anomaly:.2f}"),
            ('T',  f"{self.orbital_period:.2f}"),
            ('n',  f"{self.mean_motion:.2f}"),
            ('τ',  f"{self.time_periapsis:.2f}")
        ]
        return ' '.join([ f"[{a}={b}]" for a, b in fields ])
