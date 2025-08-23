import atexit
from collections import OrderedDict
from multiprocessing import shared_memory
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.ui.fmt import mag_format

fp = np.float64

ST_NOMINAL   = 0
ST_COLLIDING = 1
ST_COLLIDED  = 2

FIELDS = OrderedDict()

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
    'bytesize': N * np.dtype(fp).itemsize,
    'dtype': fp,
    'shape': (N,)
}

# Radius
FIELDS['radius'] = lambda N: {
    'bytesize': N * np.dtype(fp).itemsize,
    'dtype': fp,
    'shape': (N,)
}

# Distance between two bodies
FIELDS['distance'] = lambda N: {
    'bytesize': N**2 * np.dtype(fp).itemsize,
    'dtype': fp,
    'shape': (N, N)
}

# Interaction state between two bodies
FIELDS['state'] = lambda N: {
    'bytesize': N**2 * np.dtype(np.int8).itemsize,
    'dtype': np.uint8,
    'shape': (N, N)
}


class OrbitalMemory:
    _shm:shared_memory.SharedMemory
    N:int
    
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
    state:NDArray[np.uint8]
    
    def __init__(self, N:int, name:str|None=None):
        self.N = N
        self.buffer_size = sum([
            f(N)['bytesize']
            for f in FIELDS.values()
        ])
        
        if name is not None:
            self._shm = shared_memory.SharedMemory(name=name)
        else:
            # Create shared memory block
            self._shm = shared_memory.SharedMemory(create=True, size=self.buffer_size)
            atexit.register(lambda: self._shm.unlink())
            print(f"Allocated {mag_format(self.buffer_size)}B of shared memory.")

        offset = 0
        for field, f in FIELDS.items():
            f = f(N)
            setattr(self, field, np.ndarray(f['shape'], dtype=f['dtype'], buffer=self._shm.buf, offset=offset))
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
    