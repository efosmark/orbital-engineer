import math
import random

from orbitalengineer.engine.body import OrbitalBody
from orbitalengineer.engine.compute import r_from_mass, vis_viva_velocity_vector



def random_color() -> tuple[float, float, float]:
    """Create a random color in the form of (r, g, b, a).
    
    Each color will have a value between 0.2 and 1.0.
    The alpha channel will always be 1.0.

    Returns:
        tuple[float, float, float, float]: The RGBA color tuple.
    """
    red = 0.2 + (random.random() * 0.8)
    green = 0.2 + (random.random() * 0.8)
    blue = 0.2 + (random.random() * 0.8)
    return (red, green, blue)

def rand_point_annulus(r_min, r_max):
    u = random.random()
    r = math.sqrt(u * (r_max*r_max - r_min*r_min) + r_min*r_min)
    theta = random.uniform(0.0, 2.0*math.pi)
    return (r*math.cos(theta), r*math.sin(theta), r)

def _random_xy(min_distance, max_distance):
    #return rand_point_annulus(min_distance, max_distance)
    r = math.sqrt(random.uniform(min_distance**2, max_distance**2))
    theta = random.uniform(0, 2.0 * math.pi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x, y, r)

def create_primary(*, radius:float|None=None, mass:float = 332000) -> OrbitalBody:
    return OrbitalBody(
        x=0,
        y=0,
        mass=mass,
        vx=0,
        vy=0,
        radius=r_from_mass(mass) if radius is None else radius,
    )

def create_secondary(primary_body:OrbitalBody, *, min_radius:float=1, max_radius=1e5, prograde:bool=True, color=None, mass:float=0, ecc:float|None=None):
    x, y, dist = _random_xy(primary_body.radius + min_radius, max_radius )
    x, y = primary_body.x + x, primary_body.y + y
    vx, vy = vis_viva_velocity_vector(
        x,
        y,
        dist,
        primary_body.mass,
        a=dist if ecc is None else dist * ecc,
        prograde=prograde
    )
    return OrbitalBody(
        x, 
        y,
        mass,
        primary_body.vx+vx,
        primary_body.vy+vy,
        r_from_mass(mass),
    )

Color_T = tuple[float, float, float, float]

def create_random_body(*,
    x:float|None = None,
    y:float|None = None,
    dist:tuple[float,float]|None = None,
    mass:float|None = None,
    radius:float|None = None,
    vx:float|None = None,
    vy:float|None = None,
):
    if dist is None and (x is None or y is None):
        raise ValueError("Either both x and y need to be specified, or dist range.")

    if mass is None:
        mass = random.randint(1, 1000)
    
    if radius is None:
        radius = r_from_mass(mass)
    
    if dist is not None:
        x, y, _ = _random_xy(dist[0], dist[1])
    else:
        rx, ry, _ = _random_xy(0, radius * 10.0)
        if x is None:
            x = rx 
        if y is None:
            y = ry 

    if vx is None:
        vx = random.uniform(-10, 10)
    
    if vy is None:
        vy = random.uniform(-10, 10)
        
    return OrbitalBody(
        x, y,
        radius,
        vx,
        vy, 
        mass
    )
    

def binary_primary(x, y, mass, dist, prograde=False):
    x = dist/2
    y = 0
    b_radius = r_from_mass(mass/2.0)
    
    vx, vy = vis_viva_velocity_vector(-x, y, dist, mass*2, dist, prograde=prograde)
    b1 = OrbitalBody(x, y, b_radius, vx, vy, mass/2)
    
    vx, vy = vis_viva_velocity_vector(x, y, dist, mass*2, dist, prograde=prograde)
    b2 = OrbitalBody(-x, y, b_radius, vx, vy, mass/2)
    
    sol = OrbitalBody(0, 0, mass=mass, radius=r_from_mass(mass), vx=0, vy=0)
    return b1, b2, sol