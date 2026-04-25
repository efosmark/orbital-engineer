import numpy as np
from numpy.typing import NDArray
from numba import njit, void, float32, complex128, int64
from orbitalengineer.engine.orbitalnp.integrator.nudge import nudge
from orbitalengineer.engine.orbitalnp.integrator.accel import accel
from orbitalengineer.engine.orbitalnp.integrator.conf import GRAV_DEFAULT, NJIT_CACHE, NJIT_FASTMATH

@njit(void(
    int64,          # i
    int64,          # j
    complex128[:],  # velocity
    float32[:],     # radius
    complex128[:],  # position
    float32,        # CoR
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def bounce(
    i: int,
    j: int,
    velocity: NDArray[np.complex128],
    mass: NDArray[np.float32],
    position: NDArray[np.complex128],
    e:float, # coefficient of restitution
):
    dz = position[i] - position[j]
    dist = abs(dz) + 1e-10

    n = dz / dist  # unit normal from j -> i

    rv = velocity[i] - velocity[j]

    # relative speed along normal (scalar)
    vn = (rv * n.conjugate()).real

    # if already separating, no response
    if vn >= 0.0: n = 0

    inv_mass_sum = (1.0 / mass[i]) + (1.0 / mass[j])
    impulse = -(1.0 + e) * vn / inv_mass_sum  # scalar impulse magnitude

    # apply along normal (complex direction n)
    velocity[i] += (impulse / mass[i]) * n
    velocity[j] -= (impulse / mass[j]) * n


@njit(void(
    int64[:],       # body_ids
    float32,        # step size
    complex128[:],  # velocity
    float32[:],     # radius
    complex128[:],  # position
    float32[:],     # radius
    #int64[:],       # status
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def kick(
    body_ids: NDArray[np.int64],
    dt_step: np.float32,
    velocity: NDArray[np.complex128],
    mass: NDArray[np.float32],
    position: NDArray[np.complex128],
    radius: NDArray[np.float32],
):
    e = 0.97
    N = mass.size
    idx = 0
    
    overlapping_i = np.empty(N, dtype=np.uint)
    overlapping_j = np.empty(N, dtype=np.uint)
    for i in body_ids:
        for j in range(N):
            if abs(position[j] - position[i]) <= radius[i] + radius[j]:
                overlapping_i[idx] = i
                overlapping_j[idx] = j
                idx += 1
                
    for i in body_ids:
        i = int(i)
        for j in range(N):
            velocity[i] = velocity[i] + accel(mass[i], mass[j], position[j] - position[i], GRAV_DEFAULT) * dt_step
                
    for oidx in range(idx):
        i = overlapping_i[oidx]
        j = overlapping_j[oidx]
        nudge(i, j, mass, radius, position)
        bounce(i, j, velocity, mass, position, e)


# if __name__ == "__main__":
#     from orbitalengineer.engine.integrator.simd_inspect import show_simd_asm
#     show_simd_asm(kick.inspect_asm(), head=300) 