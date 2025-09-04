import atexit
from collections import OrderedDict
import math
from multiprocessing import shared_memory
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.ui.fmt import mag_format

HISTORY_DEPTH = np.int64(10)

STATUS_NOMINAL = 0
STATUS_DELETING = 1
STATUS_DELETED = 2
STATUS_COLLIDING = 3

INTERACTION_NONE   = 0
INTERACTION_COLLIDING = 1
INTERACTION_COLLISION  = 2

def offset(step_id):
    return np.int64(step_id % HISTORY_DEPTH)


FIELDS = OrderedDict()

# Step info [step_id, num_steps, dt]
FIELDS['step'] = lambda N: {
    'dtype': np.float64,
    'shape': (3,)
}

# Overall status of the body (0=normal, 1=deleted)
# FIELDS['status'] = lambda N: {
#     'dtype': np.int64,
#     'shape': (N,)
#}

# Absolute positon
#FIELDS['position'] = lambda N: {
#    'dtype': np.complex128,
#    'shape': (N,)
#}

# Absolute velocity
#FIELDS['velocity'] = lambda N: {
#    'dtype': np.complex128,
#    'shape': (N,)
#}

# # Mass
# FIELDS['mass'] = lambda N: {
#     'dtype': np.float64,
#     'shape': (N,)
# }

# Radius
# FIELDS['radius'] = lambda N: {
#     'dtype': np.float64,
#     'shape': (N,)
# }

# Distance between two bodies
FIELDS['distance'] = lambda N: {
    'dtype': np.float64,
    'shape': (N, N)
}

# Interaction state between two bodies
FIELDS['interaction'] = lambda N: {
    'dtype': np.int64,
    'shape': (N, N)
}

#
# Hitory params.
# Each is indexed via `step_id % MAX_HISTORY_DEPTH`
#
FIELDS['history_tick_id'] = lambda N: {
    'dtype': np.int64,
    'shape': (HISTORY_DEPTH,)
}
FIELDS['history_dt_step'] = lambda N: {
    'dtype': np.float64,
    'shape': (HISTORY_DEPTH,)
}
FIELDS['status'] = lambda N: {
    'dtype': np.int64,
    'shape': (HISTORY_DEPTH, N)
}
FIELDS['velocity'] = lambda N: {
    'dtype': np.complex128,
    'shape': (HISTORY_DEPTH, N)
}
FIELDS['position'] = lambda N: {
    'dtype': np.complex128,
    'shape': (HISTORY_DEPTH, N)
}
FIELDS['mass'] = lambda N: {
    'dtype': np.float64,
    'shape': (HISTORY_DEPTH, N)
}
FIELDS['radius'] = lambda N: {
    'dtype': np.float64,
    'shape': (HISTORY_DEPTH, N)
}

def get_size(field):
    return math.prod([
        np.dtype(field["dtype"]).itemsize * axis_size
        for axis_size in field["shape"]
    ])

class OrbitalMemory:
    _shm:shared_memory.SharedMemory
    N:int
    
    #
    step:NDArray[np.float64]
    
    #
    status:NDArray[np.int64]
    
    # Position in the complex plane (X+Yj)
    position:NDArray[np.complex128]
    
    # Velocity in the complex plane
    velocity:NDArray[np.complex128]
    
    # Mass (arbitrary units)
    mass:NDArray[np.float64]
    
    # Body radius (arbitrary units)
    radius:NDArray[np.float64]
    
    # Distance between two bodies.
    # Shape (N, N)
    distance:NDArray[np.float64]
    
    # The state of the interaction between bodies.
    # Shape (N, N)
    # See the constants at the top of this file for more info.
    interaction:NDArray[np.int64]
    
    history_tick_id:NDArray[np.int64]
    history_dt_step:NDArray[np.float64]
    
    def __init__(self, N:int, name:str|None=None, second_buffer:bool=False):
        self.N = N
        self.second_buffer = second_buffer
        self.buffer_size = sum([
            get_size(field)
            for field in [
                f(N) for f in FIELDS.values()
            ]
        ])
        
        if name is not None:
            self._shm = shared_memory.SharedMemory(name=name)
        
        else:
            # Create shared memory block
            self._shm = shared_memory.SharedMemory(create=True, size=(self.buffer_size * 2))
            atexit.register(lambda: self._shm.unlink())
            logger.info(f"Allocated {mag_format(self.buffer_size)}B of shared memory.")

        # Zero out the fields
        offset = 0 if not second_buffer else self.buffer_size
        for field, f in FIELDS.items():
            f = f(N)
            setattr(self, field, np.ndarray(f['shape'], dtype=f['dtype'], buffer=self._shm.buf, offset=offset))
            if name is None:
                getattr(self, field).fill(0)
            offset += get_size(f)

    def get_body_mass(self, step_id:int, body_id:int) -> np.float64:
        return self.mass[offset(step_id), body_id]

    def get_body_radius(self, step_id:int, body_id:int) -> np.float64:
        return self.radius[offset(step_id), body_id]


class BodyProxy:
    """ Convenience class for accessing body properties.

        This should not be used for bulk operations, but rather for cases when
        dealing with one or two specific bodies.
    """
    
    instances:ClassVar[dict[int, 'BodyProxy']] = dict()
    
    idx:int
    shm:OrbitalMemory
    
    def __new__(cls, idx:int, shm:OrbitalMemory):
        if idx not in cls.instances:
            self = super().__new__(cls)
            self.shm = shm
            self.idx = idx
            cls.instances[idx] = self
        return cls.instances[idx]

    def get_position(self, step_id:int):
        return self.shm.position[offset(step_id), self.idx]
    
    def get_xy(self, step_id:int) -> tuple[float,float]:
        pos = self.shm.position[offset(step_id), self.idx]
        return pos.real, pos.imag

    def get_velocity(self, step_id:int):
        return self.shm.velocity[offset(step_id), self.idx]

    def get_mass(self, step_id:int):
        return self.shm.mass[offset(step_id), self.idx]

    def get_radius(self, step_id:int):
        return self.shm.radius[offset(step_id), self.idx]
