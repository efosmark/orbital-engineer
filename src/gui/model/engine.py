from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
from gui.gtk4 import GObject

from orbitalengineer import dtypes, body
from orbitalengineer.engine.device import DeviceStatus
from orbitalengineer.engine.primary_vectors import HasStateVectors
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.ipc import message

_TENTHS = [v/10.0 for v in range(10)]
_TENS = [
    *[v for v in _TENTHS if v > 0],
    *[(1 + v) for v in _TENTHS],
    *[(2 + v) for v in _TENTHS],
    *[(3 + v) for v in _TENTHS],
    *[(4 + v) for v in _TENTHS],
    *[(5 + v) for v in _TENTHS],
    *[(6 + v) for v in _TENTHS],
    *[(7 + v) for v in _TENTHS],
    *[(8 + v) for v in _TENTHS],
    *[(9 + v) for v in _TENTHS],
]
SPEED_SCALE = [
    *[(v+1)/100.0 for v in range(9)],
]
for i in range(10):
    SPEED_SCALE.extend([t + (i*10) for t in _TENS])


class EngineModel(GObject.GObject): 
    
    shift_vector:GObject.Signal = GObject.Signal(name='shift-vector', arg_types=(object,),)
    
    config:SimConfig = GObject.Property(type=object) #type:ignore
    
    tick_id:int = GObject.Property(type=int, default=0) #type:ignore
    dt_step:float = GObject.Property(type=float, default=0) #type:ignore
    is_initialized:bool = GObject.Property(type=bool, default=False) #type:ignore
    N:int = GObject.Property(type=int, default=0) #type:ignore
    accum:float = GObject.Property(type=float, default=0) #type:ignore
    valid_indices:NDArray[np.uint32] = GObject.Property(type=object) #type:ignore
    
    clock_running:bool = GObject.Property(type=bool, default=False) #type:ignore
    clock_speed:float = GObject.Property(type=float, default=1.0) #type:ignore
    clock_time:float = GObject.Property(type=float, default=0.0) #type:ignore
    curr_tick_at:float = GObject.Property(type=float, default=0.0) #type:ignore
    next_tick_at:float = GObject.Property(type=float, default=0.0) #type:ignore

    id_to_index:NDArray[np.uint32] = GObject.Property(type=object) #type:ignore
    body_id:NDArray[np.uint32] = GObject.Property(type=object) #type:ignore
    flags:NDArray[np.uint32] = GObject.Property(type=object) #type:ignore
    position:NDArray[np.complex64|np.complex128] = GObject.Property(type=object) #type:ignore
    velocity:NDArray[np.complex64|np.complex128] = GObject.Property(type=object) #type:ignore
    mass:NDArray[np.float32] = GObject.Property(type=object) #type:ignore
    radius:NDArray[np.float32] = GObject.Property(type=object) #type:ignore
    force:NDArray[np.complex64] = GObject.Property(type=object) #type:ignore
    ledger:NDArray = GObject.Property(type=object) #type:ignore
    cgroup:NDArray[np.uint32] = GObject.Property(type=object) #type:ignore
    n_direct_contacts:NDArray = GObject.Property(type=object) #type:ignore
    direct_contacts:NDArray = GObject.Property(type=object) #type:ignore
    kinetic_energy:NDArray = GObject.Property(type=object) #type:ignore
    grav_potential_energy:NDArray = GObject.Property(type=object) #type:ignore
    
    buffer_id = GObject.Property(type=str)

    gpu_status:DeviceStatus = GObject.Property(type=object) #type:ignore
    cpu_status:message.HostStatus = GObject.Property(type=object) #type:ignore

    def __init__(self):
        super().__init__()
        self.valid_indices = np.ndarray([], dtype=np.uint32)

    def increase_speed(self):
        idx = SPEED_SCALE.index(self.clock_speed) + 1
        if idx >= len(SPEED_SCALE):
            idx = len(SPEED_SCALE) - 1
        self.clock_speed = SPEED_SCALE[idx]

    def decrease_speed(self):
        idx = SPEED_SCALE.index(self.clock_speed) - 1
        if idx < 0:
            idx = 0
        self.clock_speed = SPEED_SCALE[idx]

    def pause(self):
        self.clock_running = False
    
    def unpause(self):
        self.clock_running = True

    def __iter__(self):
        for i in self.valid_indices:
            yield self.get_particle(int(i))

    def get_particle(self, particle_id:int) -> body.Body:
        return body.Body(vec=cast(HasStateVectors, self), index=particle_id)

    def find_bodies_at(self, x:float, y:float, margin:float=10):
        if not self.is_initialized:
            return np.empty(0, dtype=np.uint32)
        
        indices = self.valid_indices
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.radius[indices] + margin

        # Relative difference between the click and every location
        p = self.position[indices]
        d = dtypes.COMPLEX(x, y) - p
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius)
        
        # Get the indices 
        return indices[mask]

    def rel_position(self, ids:Sequence[int], offset:complex):
        if not self.is_initialized: return False
        self.emit('shift-vector', message.ShiftVectorsRequest(
            vector_name='position',
            ids=ids,
            op="add",
            offset=(offset.real, offset.imag)
        ))