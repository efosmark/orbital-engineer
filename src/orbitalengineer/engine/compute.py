import math

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

def compute_distance(x1, y1, x2, y2):
    """Get the absolute distance between two points.

    Args:
        a (Point): First point
        b (Point): Second point

    Returns:
        float: The magnitude, representing the total distance.
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return 0, 0, 0
    return (dx/dist), (dy/dist), dist

def r_from_mass(m):
    r = math.sqrt(m / math.pi) #* 4
    if r < 0:
        return 1.0
    return r 

def gravitational_parameter(G:float, M:float, m:float=0):
    return G * (M + m)

def orbital_energy(mu, r, v):
    """Energy per unit mass of the orbiting body.

    Args:
        mu (float): Standard gravitational parameter
        r (float): Current distance from central body
        v (float): Relative velocity
    """
    return (v**2 / 2) - (mu / r)

def semi_major_axis(mu, oe):
    """Size of the orbit (half of major axis)"""
    return - (mu / (2 * oe))

def specific_angular_momentum(x, y, vx, vy):
    """Specific angular momentum (scalar in 2D) in m**2/s"""
    return (x * vy) - (y * vx)

def eccentricity_vector(vx, vy, h, mu, r, x, y):
    """Eccentricity vector"""
    h = specific_angular_momentum(x, y, vx, vy)
    ex = (vy * h)/mu - (x/r)
    ey = -(vx * h)/mu - (y/r)
    return ex, ey
    
def solve_kepler(M, e, tol=1e-6, max_iter=10):
    E = M
    for _ in range(max_iter):
        delta = E - e * math.sin(E) - M
        if abs(delta) < tol:
            break
        E -= delta / (1 - e * math.cos(E))
    return E