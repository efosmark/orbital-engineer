from numpy.typing import NDArray
import numpy as np
from numba import njit, prange
from orbitalengineer.engine.memory import ST_COLLIDING, ST_COLLIDED, ST_NOMINAL, fp

ENABLE_PARALLEL = True


@njit(parallel=False)
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
        
        if state[i,j] != state[j,i] != ST_NOMINAL:
            continue
        
        if distance[i,j] < (radius[i] + radius[j]):
            state[i,j] = state[j,i] = ST_COLLIDING
            
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
            state[big, small] = ST_COLLIDED
            
            v[small] = 0
            mass[small] = 0
            radius[small] = 0
            state[small, big] = ST_COLLIDED

@njit(parallel=ENABLE_PARALLEL)
def compute_accel(
    ix: NDArray, 
    jx: NDArray,
    a: NDArray[np.complex128],
    r: NDArray[np.complex128],
    mass: NDArray,
    distance: NDArray[np.float64]
):
    a.fill(0)
    for k in prange(ix.size):
        i = ix[k]
        j = jx[k]
                
        # https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
        # TODO move this to a separate njit
        d = r[j] - r[i]
        dist = np.abs(d)
        F = -1.0 * ((mass[i] * mass[j]) / (dist**3)) * d
        
        if not mass[i] or not mass[j]:
            continue
        
        a[i] -= F / mass[i]
        a[j] += F / mass[j]
        
        distance[i,j] = dist
        distance[j,i] = dist


@njit(parallel=ENABLE_PARALLEL)
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
    for k in prange(ix.size):
        i, j = ix[k], jx[k]
        v[i] += a[i] * dt_half_step
        v[j] += a[j] * dt_half_step    


@njit(parallel=ENABLE_PARALLEL)
def drift(
    ix: NDArray,
    jx: NDArray,
    v: NDArray[np.complex128],
    r: NDArray[np.complex128],
    dt_step: np.float64
):
    for k in prange(ix.size):
        i, j = ix[k], jx[k]
        r[i] += v[i] * dt_step
        r[j] += v[j] * dt_step


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
    dt_step: np.float64
):
    half_step = fp(dt_step / 2.0)

    kick(ix, jx, v, a, r, mass, distance, half_step)
    drift(ix, jx, v, r, dt_step)
    kick(ix, jx, v, a, r, mass, distance, half_step)

    compute_collisions(ix, jx, v, radius, mass, distance, state)    
