import atexit
from collections import OrderedDict
from multiprocessing import shared_memory
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.ui.fmt import mag_format


STATUS_NOMINAL = 0
STATUS_DELETED = 1

INTERACTION_NONE   = 0
INTERACTION_COLLIDING = 1
INTERACTION_COLLIDED  = 2

FIELDS = OrderedDict()

# Step info [step_id, dt]
FIELDS['step'] = lambda N: {
    'bytesize': 2 * np.dtype(np.float64).itemsize,
    'dtype': np.float64,
    'shape': (2,)
}

# Overall status of the body (0=normal, 1=deleted)
FIELDS['status'] = lambda N: {
    'bytesize': N * np.dtype(np.uint8).itemsize,
    'dtype': np.uint8,
    'shape': (N,)
}

# Absolute positon
FIELDS['r'] = lambda N: {
    'bytesize': N * np.dtype(np.complex128).itemsize,
    'dtype': np.complex128,
    'shape': (N,)
}

# Absolute velocity
FIELDS['v'] = lambda N: {
    'bytesize': N * np.dtype(np.complex128).itemsize,
    'dtype': np.complex128,
    'shape': (N,)
}

# Absolute acceleration
FIELDS['a'] = lambda N: {
    'bytesize': N * np.dtype(np.complex128).itemsize,
    'dtype': np.complex128,
    'shape': (N,)
}

# Mass
FIELDS['mass'] = lambda N: {
    'bytesize': N * np.dtype(np.float64).itemsize,
    'dtype': np.float64,
    'shape': (N,)
}

# Radius
FIELDS['radius'] = lambda N: {
    'bytesize': N * np.dtype(np.float64).itemsize,
    'dtype': np.float64,
    'shape': (N,)
}

# Distance between two bodies
FIELDS['distance'] = lambda N: {
    'bytesize': N**2 * np.dtype(np.float64).itemsize,
    'dtype': np.float64,
    'shape': (N, N)
}

# Interaction state between two bodies
FIELDS['interaction'] = lambda N: {
    'bytesize': N**2 * np.dtype(np.uint8).itemsize,
    'dtype': np.uint8,
    'shape': (N, N)
}

class OrbitalMemory:
    _shm:shared_memory.SharedMemory
    N:int
    
    #
    step:NDArray[np.float64]
    
    #
    status:NDArray[np.uint8]
    
    # Position in the complex plane (X+Yj)
    r:NDArray[np.complex128]
    
    # Velocity in the complex plane
    v:NDArray[np.complex128]
    
    # Acceleration in the complex plane
    a:NDArray[np.complex128]
    
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
    interaction:NDArray[np.uint8]
    
    def __init__(self, N:int, name:str|None=None, second_buffer:bool=False):
        self.N = N
        self.second_buffer = second_buffer
        self.buffer_size = sum([
            field['bytesize']
            for field in [
                f(N) for f in FIELDS.values()
            ]
        ])
        
        if name is not None:
            logger.info("Connecting to existing shared memory '%s' (%s buffer)", name, 'second' if second_buffer else 'first')
            self._shm = shared_memory.SharedMemory(name=name)
        
        else:
            # Create shared memory block
            self._shm = shared_memory.SharedMemory(create=True, size=(self.buffer_size * 2))
            atexit.register(lambda: self._shm.unlink())
            logger.info(f"Allocated {mag_format(self.buffer_size)}B of shared memory.")

        offset = 0 if not second_buffer else self.buffer_size
        for field, f in FIELDS.items():
            f = f(N)
            setattr(self, field, np.ndarray(f['shape'], dtype=f['dtype'], buffer=self._shm.buf, offset=offset))
            if name is None:
                getattr(self, field).fill(0)
            offset += f['bytesize']


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
    
    @property
    def x(self):
        return float(self.shm.r[self.idx].real)
    
    @property
    def y(self):
        return float(self.shm.r[self.idx].imag)

    @property
    def vx(self):
        return float(self.shm.v[self.idx].real)

    @property
    def vy(self):
        return float(self.shm.v[self.idx].imag)

    @property
    def mass(self):
        return float(self.shm.mass[self.idx])

    @property
    def radius(self):
        return float(self.shm.radius[self.idx])
    