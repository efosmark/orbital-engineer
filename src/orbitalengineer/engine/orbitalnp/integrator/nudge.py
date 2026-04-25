import numpy as np
from numba import njit, void, float32, complex128, int64
from numpy.typing import NDArray

from orbitalengineer.engine.orbitalnp.integrator.conf import NJIT_CACHE, NJIT_FASTMATH

@njit(void(
    int64,           # i
    int64,           # j
    float32[:],      # mass
    float32[:],      # radius
    complex128[:],   # position
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def nudge(
    i: int,
    j: int,
    mass: NDArray[np.float32],
    radius: NDArray[np.float32],
    position: NDArray[np.complex128],
):
    dz = position[i] - position[j]
    dist = abs(dz) + 1e-10
    
    overlap = radius[i] + radius[j] - dist
    #if overlap > 0 and i < j:
    # split correction by inverse masses (heavier moves less)
    # NUDGE
    
    inv1, inv2 = 1.0/mass[i], 1.0/mass[j]
    k = overlap / (inv1 + inv2 + 1e-10)
    n = dz / dist
    
    position[i] += complex(n.real * k*inv1, n.imag * k *inv1)
    position[j] -= complex(n.real * k*inv2, n.imag * k *inv2)