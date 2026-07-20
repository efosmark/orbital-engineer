import socket
from multiprocessing import shared_memory
from typing import Any, Sequence, cast
from dataclasses import fields

import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.engine import config
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.config import SERVER_IPC_HOST, SERVER_IPC_PORT
from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.particle_cl import ParticleCL
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ipc import transport


class ClientSocketConnection:
    _shared:dict = {}
    
    is_initialized:bool = False
    tick_id:int = 0

    accum:float = 0
    dt_base:float = config.DEFAULT_DT_BASE
    N:int = 0
    G:float = config.DEFAULT_G
    coef_of_restitution:float = config.COEF_OF_RESTITUTION
    EPS_DIST:float = config.EPS_DIST
    EPS_TIME:float = config.EPS_TIME
    
    flags:NDArray[np.uint32]
    position:NDArray[np.complex64]
    velocity:NDArray[np.complex64]
    mass:NDArray[np.float32]
    radius:NDArray[np.float32]
    force:NDArray[np.complex64]
    
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((SERVER_IPC_HOST, SERVER_IPC_PORT))
        print(f"Connected to {(SERVER_IPC_HOST, SERVER_IPC_PORT)}")
        self.clock = SimClock()
        self._prev = None
        self._uninitialized_bodies:list[transport.ParticleInit] = []

    def get_valid_indices(self) -> NDArray:
        if not self.is_initialized:
            return np.empty(0, dtype=np.uint32)
        return np.where((self.flags & flags.REMOVED) != flags.REMOVED)[0]

    def find_bodies_at(self, x:float, y:float, margin:float=10):
        indices = self.get_valid_indices()
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.radius[indices] + margin

        # Relative difference between the click and every location
        p = self.position[indices]
        d = np.complex128(x, y) - p
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius)
        
        # Get the indices 
        return indices[mask]

    def add_particle(self, position:complex, velocity:complex, mass:float, radius:float, flags:int=0):
        if self.is_initialized:
            logger.warning("Cannot add a particle once the simulation has started.")
            return -1
        p = transport.ParticleInit(
            flags=flags,
            position=(position.real, position.imag),
            velocity=(velocity.real, velocity.imag),
            mass=mass,
            radius=radius
        )
        self._uninitialized_bodies.append(p)
        return len(self._uninitialized_bodies) - 1

    def get_particle(self, particle_id:int) -> Particle:
        return ParticleCL(particle_id, self)

    def __iter__(self):
        for i in self.get_valid_indices():
            yield self.get_particle(int(i))
        
    def _apply_config(self, config: transport.ConfigResponse):
        self.N = config.N
        self.G = config.G
        self.coef_of_restitution = config.coef_of_restitution
        self.dt_base = config.dt_base
        self.EPS_DIST = config.EPS_DIST
        self.EPS_TIME = config.EPS_TIME
        logger.info("Config applied: %s", config)

    def _connect_shared_memory(self, shared: transport.SharedMemoryResponse):
        for f in fields(transport.SharedMemoryResponse):
            shm_state = cast(transport.SharedMemoryInfo, getattr(shared, f.name))
            shm = shared_memory.SharedMemory(name=shm_state.name, size=shm_state.size, track=False)            
            self._shared[f.name] = shm
            setattr(self, f.name, np.ndarray(shm_state.shape, dtype=shm_state.dtype, buffer=shm.buf))
            logger.info("Connected memory %s %s %s %s %s", f.name, shm_state.name, shm_state.size, shm_state.dtype, shm_state.shape)

    def send_message(self, message_enum: transport.MessageType, message:Any|None=None) -> bool:
        if message_enum != transport.MessageType.SYNC:
            logger.info("SEND %s", message_enum.name)
        transport.send_message(self.s, message_enum, message)
        return self._handle_response()

    def _handle_response(self) -> bool:
        _, message_type, payload = transport.recv_message(self.s, transport.MessageType)
        
        if message_type == transport.MessageType.SUCCESS:
            return True
         
        elif message_type == transport.MessageType.INIT:
            req = transport.InitResponse.from_dict(payload)
            self.is_initialized = req.initialized
            self._apply_config(req.config)
            self._connect_shared_memory(req.memory)
            return self.is_initialized
        
        elif message_type == transport.MessageType.STATUS:
            req = transport.StatusResponse.from_dict(payload)
            self.is_initialized = req.initialized
            self.tick_id = req.tick_id
            self.accum = req.accum
            self.clock.update(req.clock)
            return True
    
        elif message_type == transport.MessageType.ERROR:
            req = transport.ErrorResponse(**payload)
            logger.error(req.error_message)
            # TODO: Emit the error so it can be displayed by the interface
            return False
        
        return False

    
    def init_sim(self, platform_id:int, device_id:int):
        logger.info("Initializing sim...")
        self.platform_id = platform_id
        self.device_id = device_id
        result = self.send_message(
            transport.MessageType.INIT,
            transport.InitRequest(
                particles=self._uninitialized_bodies,
                platform_id=self.platform_id,
                device_id=self.device_id
            ))
        logger.info("Initializing sim completed with result: %s", result)
        return result
    
    def set_clock_speed(self, speed):
        return self.send_message(
            transport.MessageType.CLOCK_SET_SPEED,
            transport.ClockSetSpeedRequest(speed=speed)
        )
    
    def start(self):
        self.send_message(transport.MessageType.CLOCK_START)
    
    def stop(self):
        self.send_message(transport.MessageType.CLOCK_PAUSE)
    
    def sync(self):
        self.send_message(transport.MessageType.SYNC)

    def rel_move(self, ids:Sequence[int], offset:complex):
        self.send_message(
            transport.MessageType.BODY_SHIFT,
            transport.ShiftVectorsRequest(
                vector_name='position',
                ids=ids,
                op="add",
                offset=(offset.real, offset.imag)
            )
        )

    def rel_velocity(self, ids:Sequence[int], offset:complex):
        self.send_message(
            transport.MessageType.BODY_SHIFT,
            transport.ShiftVectorsRequest(
                vector_name='velocity',
                ids=ids,
                op="mul",
                offset=(offset.real, offset.imag)
            )
        )

    def rel_mass(self, ids:Sequence[int], offset:float):
        self.send_message(
            transport.MessageType.BODY_SHIFT,
            transport.ShiftVectorsRequest(
                vector_name='mass',
                ids=ids,
                op="mul",
                offset=(offset.real, 0)
            )
        )

    def to_dict(self) -> dict:
        self.sync()
        return {
            # Simulation state
            "tick_id": int(self.tick_id),
            "dt_base": float(self.dt_base),
            "N": int(self.N),
            
            # Constants
            "G": float(self.G),
            "coef_of_restitution": float(self.coef_of_restitution),
            "EPS_DIST": float(self.EPS_DIST),
            "EPS_TIME": float(self.EPS_TIME),
            
            # particle field vectors
            "flags":    [int(fl) for fl in self.flags],
            "position": [(f"{p.real:.6f}", f"{p.imag:.6f}") for p in self.position],
            "velocity": [(f"{v.real:.6f}", f"{v .imag:.6f}") for v in self.velocity],
            "mass":     [f"{m:.6f}" for m in self.mass],
            "radius":   [f"{r:.6f}" for r in self.radius],
            # force is omitted due to size and lower importance
        }
