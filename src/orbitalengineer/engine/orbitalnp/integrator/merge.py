import numpy as np
from numpy.typing import NDArray
from numba import njit, void, float64, complex128, int64


from orbitalengineer.engine.orbitalnp.integrator.com import center_of_mass_position, center_of_mass_velocity
from orbitalengineer.engine.orbitalnp.integrator.conf import MERGE_STRATEGY_COMBINE, NJIT_CACHE, NJIT_FASTMATH
from orbitalengineer.engine.orbitalnp.integrator.drift import drift
from orbitalengineer.engine.orbitalnp.integrator.accel import accel, force
from orbitalengineer.engine.orbitalnp.integrator.overlap import get_overlapping
from orbitalengineer.engine.orbitalnp.memory import INTERACTION_MERGED, INTERACTION_NEAR, INTERACTION_OVERLAP, STATUS_DELETED, STATUS_NOMINAL



@njit(void(
    int64[:],    # body_ids
    float64[:],  # mass
    float64[:],  # radius
    int64[:],    # status
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def mark_as_merged(
    body_ids: NDArray[np.int64],
    mass: NDArray[np.float64],
    radius: NDArray[np.float64],
    status: NDArray[np.int64]
):
    for ii in range(body_ids.size):
        i = body_ids[ii]
        status[i] = STATUS_DELETED
        mass[i] = 0
        radius[i] = 0
        # for jj in range(ii + 1, body_ids.size):
        #    j = body_ids[jj]
        #    interaction[i, j] = interaction[j, i] = INTERACTION_MERGED


@njit(float64(float64), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH)
def r_from_mass(m: np.float64) -> np.float64:
    r = np.sqrt(m / np.pi)
    if r <= 0:
        return np.float64(1.0)
    return r



@njit(void(
    int64[:],        # body_ids
    int64,           # group_id
    float64[:],      # mass
    complex128[:],   # velocity
    complex128[:],   # position
    float64[:],      # radius
    int64[:],        # status
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def combine_bodies(
    body_ids: NDArray[np.int64],
    group_id: np.int64,
    mass: NDArray[np.float64],
    velocity: NDArray[np.complex128],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    status: NDArray[np.int64],
):
    
    total_mass = mass[body_ids].sum()
    v_cm = center_of_mass_velocity(body_ids, mass, velocity)
    r_cm = center_of_mass_position(body_ids, mass, position)                
    mark_as_merged(body_ids, mass, radius, status)
 
    mass[group_id] = total_mass
    velocity[group_id] = v_cm
    position[group_id] = r_cm
    radius[group_id] = r_from_mass(total_mass)
    status[group_id] = STATUS_NOMINAL


@njit(void(
    int64[:],        # body_ids
    float64[:,:],    # distance
    float64[:],      # mass
    complex128[:],   # velocity
    complex128[:],   # position
    float64[:],      # radius
    int64[:],        # status
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def merge(
    body_ids: NDArray[np.int64],
    distance: NDArray[np.float64],
    mass: NDArray[np.float64],
    velocity: NDArray[np.complex128],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    status: NDArray[np.int64],
):
    ids_to_process = body_ids.copy()
    while ids_to_process.size > 0:
        i = int(ids_to_process[0])
        
        gbody_ids = get_overlapping(i, radius, status, distance)
        #gbody_ids = get_overlapping2(i, group, interaction)
        ids_to_process = np.setdiff1d(ids_to_process, gbody_ids)

        if gbody_ids.size <= 1:
            continue
        
        if mass[i] == 0: 
            continue
        
        # In order to reduce running the same computation across multiple processes,
        # this will partition it to whichever process is responsible for the lowest ID.
        min_id = gbody_ids.min()
        if min_id in body_ids:
            combine_bodies(gbody_ids, min_id, mass, velocity, position, radius, status)
    
        