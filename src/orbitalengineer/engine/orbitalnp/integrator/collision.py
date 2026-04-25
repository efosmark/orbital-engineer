import numpy as np
from numba import njit, void, float64, complex128, int64
from numpy.typing import NDArray
from orbitalengineer.engine.orbitalnp.integrator.conf import NJIT_CACHE, NJIT_FASTMATH
from orbitalengineer.engine.orbitalnp.memory import INTERACTION_NEAR, INTERACTION_OVERLAP, STATUS_NOMINAL

@njit(int64(
    int64,           # index 
    int64[:],        # status
    float64[:],      # mass
    complex128[:],   # position
    float64[:],      # radius
    int64[:,:],      # interaction
    float64[:,:],    # distance
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def find_collisions(
    i: np.int64,
    status: NDArray[np.int64],
    mass: NDArray[np.float64],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    interaction: NDArray[np.int64],
    distance: NDArray[np.float64]
) -> int:
    """Given a particle, mark any overlapping particles in the interaction matrix.
    
    Args:
        i (np.int64): The particle/body ID (essentially the array index)
        status (NDArray[np.int64]): Status array for each particle (0=nominal, 1=deleted)
        position (NDArray[np.complex128]): Complex position for each particle (x+yj)
        radius (NDArray[np.float64]): Radius for each particle
        interaction (NDArray[np.int64]): Interaction matrix for each particle pair (i,j)
    
    
    It performs in-place changes to the interaction table.
    """
    if radius[i] == 0:
        # Particles that take up no space currently have no collision physics (e.g. WIMPs)
        return 0
    count:int = 0
    for j in range(i+1, position.size):
        if status[j] != STATUS_NOMINAL or radius[j] == 0:
            continue
        distance[i,j] = distance[j,i] = abs(position[j] - position[i])
        if distance[i,j] < 1:
           distance[i,j] = distance[j,i] = 1
           
        overlap = radius[i] + radius[j] - distance[i,j]
        if overlap > 0:            # split correction by inverse masses (heavier moves less)
            inv1, inv2 = 1.0/mass[i], 1.0/mass[j]
            k = overlap / (inv1 + inv2)
            
            dz = position[i] - position[j]
            n = dz / distance[i,j]
            
            position[i] += complex(n.real * k*inv1, n.imag * k *inv1)
            position[j] -= complex(n.real * k*inv2, n.imag * k *inv2)
            
            count += 1
            interaction[i,j] = interaction[j,i] = np.int64(INTERACTION_OVERLAP)
        else:
            interaction[i,j] = interaction[j,i] = 0
    return count