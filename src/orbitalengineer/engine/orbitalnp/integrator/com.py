import numpy as np
from numpy.typing import NDArray
from numba import njit, float64, complex128, int64
from orbitalengineer.engine.orbitalnp.integrator.conf import NJIT_CACHE, NJIT_FASTMATH

@njit(complex128(
    int64[:],       # body_ids
    float64[:],     # mass
    complex128[:]   # velocity
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def center_of_mass_velocity(
    body_ids: NDArray[np.int64],
    mass: NDArray,
    velocity: NDArray
):
    momentum = mass[body_ids] * velocity[body_ids]
    total_momentum = momentum.sum()
    total_mass = mass[body_ids].sum()
    return total_momentum / (total_mass + 1e-10)


@njit(complex128(
    int64[:],      # body_ids
    float64[:],    # mass
    complex128[:]  # position
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def center_of_mass_position(
    body_ids: NDArray[np.int64],
    mass: NDArray,
    position: NDArray
) -> complex:
    mr = mass[body_ids] * position[body_ids]
    return mr.sum() / mass[body_ids].sum()
