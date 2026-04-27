import math
import numpy as np

from orbitalengineer.engine.particle import Particle, ParticleRaw

seed = 0xf00d1e1
rng = np.random.default_rng(seed)

Color_T = tuple[float, float, float, float]

def r_from_mass(m: np.float64) -> np.float64:
    r = np.cbrt(m / np.pi)
    if r <= 0:
        return np.float64(1.0)
    return r

def vis_viva(position: complex, mu: float, a: float, prograde: bool = True) -> complex:
    r = abs(position)                              # radius
    v_mag = math.sqrt(mu * (2.0/r - 1.0/a))        # vis-viva speed
    r_hat = position / r                           # unit radial (complex)
    t_hat = r_hat * (1j if prograde else -1j)      # rotate ±90° for tangent
    return v_mag * t_hat                           # velocity vector (complex)


def random_color() -> tuple[float, float, float]:
    """Create a random color in the form of (r, g, b, a).
    
    Each color will have a value between 0.2 and 1.0.
    The alpha channel will always be 1.0.

    Returns:
        tuple[float, float, float, float]: The RGBA color tuple.
    """
    return (0.2 + (rng.random(size=3) * 0.8)).tolist()

def angular_position(theta:float, r:float) -> complex:
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return complex(x, y)

def random_position(min_distance, max_distance) -> complex:
    """Generates a random x+yj point on an annulus."""
    r = math.sqrt(rng.uniform(min_distance**2, max_distance**2))
    theta = rng.uniform(0, 2.0 * math.pi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return complex(x, y)

def create_primary(*, mass:float = 332000) -> Particle:
    return ParticleRaw(
        position=0+0j,
        velocity=0+0j,
        mass=mass,
        radius=r_from_mass(np.float64(mass)),
    )

def create_secondary(
    primary_body:Particle,
    *,
    dist:float|tuple[float,float]|None=None,
    position:complex|None=None,
    prograde:bool=True,
    mass:float=0,
    ecc:float|None=None,
    radius:float|None=None
):
    if dist is not None:
        if isinstance(dist, tuple):
            min_dist, max_dist = dist[0], dist[1]
        elif dist is not None:
            min_dist = max_dist = dist
        position = random_position(primary_body.get_radius() + min_dist, primary_body.get_radius() + max_dist)
    
    elif position is not None:
        dist = abs(position)
    else:
        raise Exception("Must specify either dist or position")
    
    dist = abs(position)
    velocity = vis_viva(
        position=position,
        mu=primary_body.get_mass(),
        a=dist if ecc is None else dist * ecc,
        prograde=prograde
    )
    
    return ParticleRaw(
        position=primary_body.get_position() + position,
        velocity=primary_body.get_velocity() + velocity,
        mass=mass,
        radius=r_from_mass(np.float64(mass)) if radius is None else radius
    )