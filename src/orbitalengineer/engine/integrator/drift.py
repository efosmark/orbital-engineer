import numpy as np
from numba import njit
from numpy.typing import NDArray
from orbitalengineer.engine.memory import OVERLAPPING, STATUS_NOMINAL

@njit(cache=True, fastmath=True)
def find_collisions(
    i: np.int64,
    position_in: NDArray[np.complex128],
    radius_in: NDArray[np.float64],
    interaction_out: NDArray[np.int64]
):
    for j in np.arange(i+1, position_in.size):
        if radius_in[i] == 0 or radius_in[j] == 0:
            continue
        d = position_in[j] - position_in[i]
        dist = np.abs(d) 
        if dist < 1:
            dist = 1
        if dist < (radius_in[i] + radius_in[j]):
            interaction_out[i,j] = interaction_out[j,i] = np.int64(OVERLAPPING)
        else:
            interaction_out[i,j] = interaction_out[j,i] = 0


@njit(cache=True, fastmath=True)
def drift(
    body_ids: NDArray[np.int64],
    dt: np.float64,
    status: NDArray[np.int64],
    radius_in: NDArray[np.float64],
    velocity_in: NDArray[np.complex128],
    position_in: NDArray[np.complex128],
    position_out: NDArray[np.complex128],
    interaction_out: NDArray[np.int64]
):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        if status[i] != STATUS_NOMINAL:
            continue
        position_out[i] = position_in[i] + (velocity_in[i] * dt)
        find_collisions(i, position_out, radius_in, interaction_out)