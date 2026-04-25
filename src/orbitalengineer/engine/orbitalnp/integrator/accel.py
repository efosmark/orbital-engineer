import numpy as np
from numba import njit, float32, complex128
from orbitalengineer.engine.orbitalnp.integrator.conf import GRAV_DEFAULT, MINIMUM_MASS_SIZE, NJIT_CACHE, NJIT_FASTMATH


@njit(complex128(
    float32,     # mass_i
    float32,     # mass_j
    complex128,  # relative distance
    float32,     # G
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def force(
    mass_i: np.float32,
    mass_j: np.float32, 
    dr: complex,
    G: np.float32
) -> complex:
    dist = np.float32(abs(dr))
    if dist == 0:
        return np.complex128(0)
    return complex(G * ((mass_i * mass_j) / (dist**3)) * dr)


@njit(complex128(
    float32,     # mass_i
    float32,     # mass_j
    complex128,  # relative distance
    float32,     # G
), cache=False, fastmath=NJIT_FASTMATH)
def accel(
    mass_i: np.float32,
    mass_j: np.float32, 
    dr: complex,
    G: np.float32
) -> complex:
    """Find the acceleration between two bodies using Newtwon's Law of Universal Gravitation.
    
    See: https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
    """
    return complex(force(mass_i, mass_j, dr, G) / (mass_i + complex(1e-10, 1e-10)))
