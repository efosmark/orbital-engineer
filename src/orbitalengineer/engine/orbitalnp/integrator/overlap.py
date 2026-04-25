import numpy as np
from numpy.typing import NDArray
from numba import njit, void, float64, complex128, int64, intp
from orbitalengineer.engine.orbitalnp.integrator.conf import NJIT_CACHE, NJIT_FASTMATH
from orbitalengineer.engine.orbitalnp.memory import INTERACTION_NEAR, INTERACTION_OVERLAP


@njit(int64[:](
    int64,         # index
    float64[:],    # radius
    int64[:],      # status
    float64[:,:],  # distance 
), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH, nogil=True)
def get_overlapping(
    i:int,
    radius:NDArray[np.float64],
    status:NDArray[np.int64],
    distance:NDArray[np.float64],
):
    # Init the group with the current body and all overlapping bodies
    group = np.empty(1, dtype=np.int64)
    group[0] = i

    overlaps = np.empty(1, dtype=np.int64)
    overlaps[0] = i
    
    while overlaps.size > 0:
        j = overlaps[0]
        overlaps = overlaps[1:]
        
        for k in range(radius.size):
            if k in group or k in overlaps or status[k] != 0:
                continue
            
            if distance[j,k] <= (radius[j] + radius[k]):
                group = np.append(group, k)
                if k not in overlaps:
                    overlaps = np.append(overlaps, k)    
    return group




# @njit(intp[:](
#     int64,         # index
#     int64[:,:],    # interaction 
# ), cache=NJIT_CACHE, fastmath=NJIT_FASTMATH)
# def get_overlapping2(
#     i:np.int64,
#     interaction:NDArray[np.int64],
# ):
#     return 0
#     gbody_ids = np.empty(groups.size, dtype=np.int64)
#     gbody_ids.fill(-1)
#     gbody_ids[0] = i
#     cursor = 1
    
#     c = 0
#     while c < cursor:
#         j = gbody_ids[c]
#         if groups[j] != -1:
#             groups[i] = groups[j]
#         for k in range(groups.size):
#             if interaction[k,j] == INTERACTION_OVERLAP and k not in gbody_ids:
#                 gbody_ids[cursor] = k
#                 cursor += 1
#         c += 1
#     return gbody_ids[:cursor]
