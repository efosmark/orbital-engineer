import numpy as np
from numpy.typing import NDArray
from numba import njit, void, float64, complex128, int64

from orbitalengineer.engine.memory import MERGED, OVERLAPPING, STATUS_DELETED, STATUS_NOMINAL


@njit(complex128(int64[:], float64[:], complex128[:]), cache=True, fastmath=True)
def center_of_mass_velocity(
    body_ids: NDArray[np.int64,],
    mass: NDArray,
    velocity: NDArray
):
    momentum = mass[body_ids] * velocity[body_ids]
    total_momentum = momentum.sum()
    total_mass = mass[body_ids].sum()
    return total_momentum / (total_mass + 1e-10)


@njit(complex128(int64[:], float64[:], complex128[:]), cache=True, fastmath=True)
def center_of_mass_position(
    body_ids: NDArray[np.int64,],
    mass: NDArray,
    position: NDArray
):
    mr = mass[body_ids] * position[body_ids]
    return mr.sum() / mass[body_ids].sum()


@njit(int64[:](int64, int64[:,:]), cache=True, fastmath=True)
def get_overlapping(i:np.int64, interaction: NDArray[np.int64]):
    
    # Init the group with the current body and all overlapping bodies
    group = np.empty(1, dtype=np.int64)
    group[0] = i

    overlaps = np.argwhere(interaction[i,] == OVERLAPPING).flatten()
    while overlaps.size > 0:
        j = overlaps[0]
        overlaps = overlaps[1:]
        
        # Get the items that overlap j
        jo = np.argwhere(interaction[j] == OVERLAPPING).flatten()
        
        # Remove any items already in `group`
        jo = np.setdiff1d(overlaps, group)
                
        # Add all of the remaining overlaps to be processed
        overlaps = np.union1d(overlaps, jo)
        
        group = np.append(group, j)
    return group


@njit(void(int64[:], float64[:], float64[:], int64[:], int64[:,:]), cache=True, fastmath=True)
def mark_as_merged(
    body_ids: NDArray[np.int64],
    mass_out: NDArray[np.float64],
    radius_out: NDArray[np.float64],
    status_out: NDArray[np.int64],
    interaction_out: NDArray[np.int64]
):
    for ii in range(body_ids.size):
        i = body_ids[ii]
        mass_out[i] = 0
        radius_out[i] = 0
        status_out[i] = STATUS_DELETED
       
        for jj in range(ii+ 1, body_ids.size):
            j = body_ids[jj]
            interaction_out[i, j] = interaction_out[j, i] = MERGED


@njit(float64(float64), cache=True, fastmath=True)
def r_from_mass(m: np.float64) -> np.float64:
    r = np.sqrt(m / np.pi)
    if r <= 0:
        return np.float64(1.0)
    return r


@njit(void(int64[:], int64[:,:], float64[:], complex128[:], complex128[:], float64[:], int64[:]), cache=True, fastmath=True)
def merge(
    body_ids: NDArray[np.int64,],
    interaction: NDArray[np.int64],
    mass: NDArray[np.float64],
    velocity: NDArray[np.complex128],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    status: NDArray[np.int64]
):
    for i in body_ids:
        if mass[i] == 0:
            continue
        group = get_overlapping(i, interaction)
        
        # In order to reduce running the same computation across multiple processes,
        # this will partition it to whichever process is responsible for the lowest ID.
        min_id = group.min()
        if group.size > 1 and min_id in body_ids:

            total_mass = mass[group].sum()
            v_cm = center_of_mass_velocity(group, mass, velocity)
            r_cm = center_of_mass_position(group, mass, position)

            mark_as_merged(group, mass, radius, status, interaction)

            mass[min_id] = total_mass
            velocity[min_id] = v_cm
            position[min_id] = r_cm
            radius[min_id] = r_from_mass(total_mass)
            status[min_id] = STATUS_NOMINAL
