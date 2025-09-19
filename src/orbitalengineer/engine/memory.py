import atexit
from collections import OrderedDict
import math
from multiprocessing import shared_memory
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.engine.history import HISTORY_DEPTH
from orbitalengineer.ui.fmt import mag_format


STATUS_NOMINAL = 0
STATUS_DELETING = 1
STATUS_DELETED = 2
STATUS_COLLIDING = 3

INTERACTION_NONE = 0
OVERLAPPING = 1
MERGED  = 2

FIELDS = OrderedDict()

# Step info [step_id, num_steps, dt]
FIELDS['step_id'] = lambda N: {
    'dtype': np.int64,
    'shape': (1,)
}

FIELDS['num_steps'] = lambda N: {
    'dtype': np.int64,
    'shape': (1,)
}

FIELDS['dt'] = lambda N: {
    'dtype': np.float64,
    'shape': (1,)
}

# Last distance between bodies
# For KDK integrator, this gets written during the kick
FIELDS['distance'] = lambda N: {
    'dtype': np.int64,
    'shape': (N, N)
}

# Interaction state between two bodies
# For KDK integrator, this gets written during the drift (see: kernel/drift.py/find_collisions)
FIELDS['interaction'] = lambda N: {
    'dtype': np.int64,
    'shape': (N, N)
}

# Each is indexed via `step_id % MAX_HISTORY_DEPTH`
FIELDS['status'] = lambda N: {
    'dtype': np.int64,
    'shape': (N,)
}

FIELDS['velocity'] = lambda N: {
    'dtype': np.complex128,
    'shape': (2, N)
}

FIELDS['position'] = lambda N: {
    'dtype': np.complex128,
    'shape': (2, N)
}

FIELDS['mass'] = lambda N: {
    'dtype': np.float64,
    'shape': (2, N)
}

FIELDS['radius'] = lambda N: {
    'dtype': np.float64,
    'shape': (2, N)
}

FIELDS['history'] = lambda N: {
    'dtype': np.complex128,
    'shape': (N, HISTORY_DEPTH)
}

FIELDS['history_index'] = lambda N: {
    'dtype': np.int64,
    'shape': (N,)
}

def get_size(field):
    """Get the full byte size of a field based on its dtype and shape."""
    return math.prod([
        np.dtype(field["dtype"]).itemsize * axis_size
        for axis_size in field["shape"]
    ])

class OrbitalMemory(shared_memory.SharedMemory):
    N:int
    
    # Current master step ID
    step_id:NDArray[np.float64]
    
    # number of steps to process
    num_steps:NDArray[np.int64]
    
    # time per step
    dt:NDArray[np.float64]
    
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
    
    history:NDArray[np.complex128]
    history_index: NDArray[np.int64]
    
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
            shared_memory.SharedMemory.__init__(self, name=name)
        
        else:
            # Create shared memory block
            shared_memory.SharedMemory.__init__(self, create=True, size=(self.buffer_size * 2))
            atexit.register(lambda: self.unlink())
            logger.info(f"Allocated {mag_format(self.buffer_size)}B of shared memory.")

        # Zero out the fields
        offset = 0 if not second_buffer else self.buffer_size
        for field, f in FIELDS.items():
            f = f(N)
            setattr(self, field, np.ndarray(f['shape'], dtype=f['dtype'], buffer=self.buf, offset=offset))
            if name is None:
                getattr(self, field).fill(0)
            offset += get_size(f)

    def get_step_id(self) -> np.float64:
        return self.step_id[0]
        

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
        return self.shm.position[step_id % 2, self.idx]
    
    def get_xy(self, step_id:int) -> tuple[float,float]:
        pos = self.shm.position[step_id % 2, self.idx]
        return pos.real, pos.imag

    def get_velocity(self, step_id:int):
        return self.shm.velocity[step_id % 2, self.idx]

    def get_mass(self, step_id:int):
        return self.shm.mass[step_id % 2, self.idx]

    def get_radius(self, step_id:int):
        return self.shm.radius[step_id % 2, self.idx]

    def get_status(self, step_id:int):
        return self.shm.status[step_id % 2, self.idx]
