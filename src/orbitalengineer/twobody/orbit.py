import numpy as np

from orbitalengineer import flags
from orbitalengineer.twobody import twobody
from orbitalengineer.engine.primary_vectors import HasStateVectors

EPS = 1e-15

def get_focus(body_idx:int, N:int, force_vec, vec:HasStateVectors) -> tuple[int,int]|None:
    idx_start = body_idx * N
    idx_stop = idx_start + N
            
    # Pull the velocity for 0..N from the velocity matrix (row=self.idx)
    force = np.abs(np.nan_to_num(force_vec[idx_start:idx_stop]))
    force_total = np.sum(force)

    filter = vec.mass < vec.mass[body_idx]                   # Greater mass than self
    filter |= (vec.flags & flags.REMOVED) == flags.REMOVED   # Only valid bodies

    # zero out for any invalid bodies (ensures they never get chosen as the focus)
    force[filter] = 0.0

    # Normalize the force
    influence = np.true_divide(force, force_total+EPS)

    # Find the top acceleration
    primary = np.argmax(influence)    
    if primary is None or primary == body_idx:return
    return int(primary), influence[primary]


def get_orbit_info(secondary:int, primary:int, vec:HasStateVectors) -> twobody.TwoBody|None:
    
    if vec.mass[secondary] > vec.mass[primary]:
       secondary, primary = primary, secondary
    
    # Orbital state parameters
    r1 = (vec.position[primary].real, vec.position[primary].imag)
    r2 = (vec.position[secondary].real, vec.position[secondary].imag)
    v1 = (vec.velocity[primary].real, vec.velocity[primary].imag)
    v2 = (vec.velocity[secondary].real, vec.velocity[secondary].imag)
    m1 = vec.mass[primary]
    m2 = vec.mass[secondary]
    
    radius1 = vec.radius[primary]
    radius2 = vec.radius[secondary]
    
    try:
        o = twobody.TwoBody(r1, r2, v1, v2, m1, m2, radius1, radius2)
    except ZeroDivisionError:
        return
    
    if not o.is_bound: return
    
    o.body_1 = primary
    o.body_2 = secondary
    return o