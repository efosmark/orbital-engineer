import numpy as np
from numba import njit, void, float32, complex128, int64
from numpy.typing import NDArray

from orbitalengineer.engine.orbitalnp.integrator.conf import MERGE_STRATEGY_COMBINE, NJIT_CACHE, NJIT_FASTMATH
from orbitalengineer.engine.orbitalnp.integrator.collision import find_collisions
from orbitalengineer.engine.orbitalnp.integrator.nudge import nudge
from orbitalengineer.engine.orbitalnp.memory import STATUS_NOMINAL

@njit(void(
    int64[:],        # body_ids
    float32,         # dt
    #int64[:],        # status
    #float32[:],      # mass
    #float32[:],      # radius
    complex128[:],   # velocity
    complex128[:],   # position
    #float32[:,:],    # distance
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def drift(
    body_ids: NDArray[np.int64],
    dt: np.float32,
    #status: NDArray[np.int64],
    #mass: NDArray[np.float32],
    #radius: NDArray[np.float32],
    velocity: NDArray[np.complex128],
    position: NDArray[np.complex128],
    #distance: NDArray[np.float32],
):
    for i in body_ids:
        position[i] += (velocity[i] * dt)
        
        #find_collisions(i, status, mass, position, radius, interaction, distance)
        
        #for j in range(i+1, position.size):
            #if status[j] != STATUS_NOMINAL or radius[j] == 0:
            #    continue
        #    distance[i,j] = abs(position[j] - position[i])
        #    distance[j,i] = distance[i,j]
        #    nudge(i, j, mass, radius, position)