from numpy.typing import NDArray
import numpy as np
from numba import njit, prange
from orbitalengineer.engine.memory import INTERACTION_COLLIDING, INTERACTION_COLLIDED, INTERACTION_NONE


@njit
def compute_collisions(
    ix: NDArray, 
    jx: NDArray,
    v: NDArray[np.complex128],
    radius: NDArray[np.float64],
    mass: NDArray[np.float64],
    distance: NDArray[np.float64],
    state: NDArray[np.uint8]
):
    for k in prange(ix.size):
        i = ix[k]
        j = jx[k]
        
        if state[i,j] != state[j,i] != INTERACTION_NONE:
            continue
        
        if distance[i,j] < (radius[i] + radius[j]):
            state[i,j] = state[j,i] = INTERACTION_COLLIDING
            
            mi = mass[i]
            mj = mass[j]
            if mi == 0 or mj == 0:
                continue
            
            combined_mass = mi + mj
            
            # Combine velocity by center-of-mass momentum
            momentum_i = v[i] * mi
            momentum_j = v[j] * mj
            combined_velocity = ((momentum_i + momentum_j) / combined_mass)
            
            # Move the mass from the smaller body to the larger body
            
            big, small = i, j
            if mi < mj:
                big, small = j, i
            
            v[big] = combined_velocity
            mass[big] = combined_mass
            radius[big] = np.sqrt(mass[big] / np.pi)
            if radius[big] < 1:
                radius[big] = 1
            state[big, small] = INTERACTION_COLLIDED
            
            v[small] = 0
            mass[small] = 0
            radius[small] = 0
            state[small, big] = INTERACTION_COLLIDED

@njit
def compute_accel(
    ix: NDArray, 
    jx: NDArray,
    a_out: NDArray[np.complex128],
    r_in: NDArray[np.complex128],
    mass_in: NDArray,
    distance_out: NDArray[np.float64]
):
    a_out.fill(0)
    a_out.fill(0)
    for k in prange(ix.size):
        i = ix[k]
        j = jx[k]
                
        # https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
        d = r_in[j] - r_in[i]
        dist = np.abs(d)
        if not dist:
            continue
        F = -1.0 * ((mass_in[i] * mass_in[j]) / (dist**3)) * d
        
        if not mass_in[i] or not mass_in[j]:
            continue
        
        a_out[i] -= F / (mass_in[i] + 1e-8)
        a_out[j] += F / (mass_in[j] + 1e-8)
        
        distance_out[i,j] = dist
        distance_out[j,i] = dist

@njit
def update_velocity(
    ix: NDArray,
    jx: NDArray,
    v_in: NDArray[np.complex128],
    v_out: NDArray[np.complex128],
    a_in: NDArray[np.complex128],
    dt_half_step: np.float64
):
    for k in prange(ix.size):
        i, j = ix[k], jx[k]
        v_out[i] = v_in[i] + (a_in[i] * dt_half_step)
        v_out[j] = v_in[j] + (a_in[j] * dt_half_step)

@njit
def kick(
    ix: NDArray,
    jx: NDArray,
    v: NDArray[np.complex128],
    a: NDArray[np.complex128],
    r: NDArray[np.complex128],
    mass: NDArray[np.float64],
    distance: NDArray[np.float64],
    dt_half_step: np.float64
):
    compute_accel(ix, jx, a, r, mass, distance)
    update_velocity(ix, jx, v, v, a, dt_half_step) 


@njit
def drift(
    ix: NDArray,
    jx: NDArray,
    v_in: NDArray[np.complex128],
    r_in: NDArray[np.complex128],
    r_out: NDArray[np.complex128],
    dt_step: np.float64
):
    for k in prange(ix.size):
        i, j = ix[k], jx[k]
        r_out[i] = r_in[i] + (v_in[i] * dt_step)
        r_out[j] = r_in[j] + (v_in[j] * dt_step)


@njit
def integrator_leapfrog_kdk_njit(
    ix: NDArray,
    jx: NDArray,
    v: NDArray[np.complex128],
    r: NDArray[np.complex128],
    a: NDArray[np.complex128],
    radius: NDArray[np.float64],
    mass: NDArray[np.float64],
    distance: NDArray[np.float64],
    state: NDArray[np.uint8],
    dt_step: np.float64,
):
    half_step = np.float64(dt_step / 2.0)

    kick(ix, jx, v, a, r, mass, distance, half_step)
    drift(ix, jx, v, r, r, dt_step)
    kick(ix, jx, v, a, r, mass, distance, half_step)

    compute_collisions(ix, jx, v, radius, mass, distance, state)    
