from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalnp.history import HISTORY_DEPTH
from orbitalengineer.engine.orbitalnp import sshm
from orbitalengineer.engine.simcontroller import Particle


STATUS_NOMINAL = 0
STATUS_DELETING = 1
STATUS_DELETED = 2
STATUS_COLLIDING = 3

INTERACTION_NONE = 0
INTERACTION_OVERLAP = 1
INTERACTION_MERGED  = 2
INTERACTION_NEAR = 3
INTERACTION_ESCAPE = 4


class OrbitalMemory(sshm.StructuredSharedMemory):
    
    buffer_ids:NDArray[np.int64] = sshm.field(2, default=0)
    num_steps:NDArray[np.int64] = sshm.field(1)
    dt:NDArray[np.float32] = sshm.field(1)
    
    position:NDArray[np.complex128] = sshm.field(...)
    velocity:NDArray[np.complex128] = sshm.field(...)
    mass:NDArray[np.float32] = sshm.field(...)
    radius:NDArray[np.float32] = sshm.field(...)
    status:NDArray[np.int64] = sshm.field(...)
    
    history:NDArray[np.complex128] = sshm.field(..., HISTORY_DEPTH)
    history_index: NDArray[np.int64] = sshm.field(...,)

    def get_particle(self, idx:int) -> 'BodyProxyNP':
        return BodyProxyNP(idx, self)


class BodyProxyNP(Particle):
    """ Convenience class for accessing body properties.

        This should not be used for bulk operations, but rather for cases when
        dealing with one or two specific bodies.
    """
    
    instances:ClassVar[dict[int, 'BodyProxyNP']] = dict()
    
    idx:int
    shm:OrbitalMemory
    
    def __new__(cls, idx:int, shm:OrbitalMemory):
        if idx not in cls.instances:
            self = super().__new__(cls)
            self.shm = shm
            self.idx = idx
            cls.instances[idx] = self
            
        return cls.instances[idx]

    def get_position(self):
        return self.shm.position[self.idx]
    
    def get_xy(self) -> tuple[float,float]:
        pos = self.shm.position[self.idx]
        return pos.real, pos.imag

    def get_velocity(self):
        return self.shm.velocity[self.idx]

    def get_mass(self):
        return self.shm.mass[self.idx]

    def get_radius(self):
        return self.shm.radius[self.idx]

    def get_status(self):
        return self.shm.status[self.idx]