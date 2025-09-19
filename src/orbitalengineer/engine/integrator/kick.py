import numpy as np
from numpy.typing import NDArray
from numba import njit, void, float64, complex128, int64
from orbitalengineer.engine.memory import STATUS_DELETED

G = np.float64(1)

@njit(complex128(float64, float64, complex128), cache=True, fastmath=True)
def accel(
    mass_i: np.float64,
    mass_j: np.float64,
    dr: np.complex128
) -> np.complex128:
    """Find the acceleration between two bodies using Newtwon's Law of Universal Gravitation.
    
    See: https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
    """
    if mass_i == 0:
        return np.complex128(0)
    
    dist = np.abs(dr)
    if dist == 0:
        return np.complex128(0)
    F = 1.0 * ((mass_i * mass_j) / (dist**3)) * dr
    return F / (mass_i)


@njit(void(int64[:], float64, complex128[:], float64[:], complex128[:], float64[:], int64[:]), cache=True, fastmath=True)
def kick(
    body_ids: NDArray[np.int64],
    half_step: np.float64,
    velocity: NDArray[np.complex128],
    mass: NDArray[np.float64],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    status: NDArray[np.int64],
):
    for k in range(body_ids.size):
        i:np.int64 = body_ids[k]
        if status[i] == STATUS_DELETED:
            continue
        for j in range(mass.size):
            if i == j or status[j] == STATUS_DELETED:
                continue
            
            d = position[j] - position[i]
            if (radius[i] == 0 or radius[j] == 0) and np.abs(d) <= 1:
                continue
            
            velocity[i] = velocity[i] + accel(mass[i], mass[j], d) * half_step
