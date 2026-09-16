import socket
from multiprocessing import shared_memory
from typing import Any, Sequence, cast
from dataclasses import asdict, fields

import numpy as np
from numpy.typing import NDArray

from orbitalengineer import flags
from orbitalengineer.engine import config, logger
from orbitalengineer.engine.orbitalcl.device import GPUStatus
from orbitalengineer.engine.orbitalcl.particle_cl import ParticleCL
from orbitalengineer.engine.orbitalcl.sim_config import SimConfig
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ipc import message, transport
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.config import SERVER_IPC_HOST, SERVER_IPC_PORT


class ClientSocketConnection:
    _shared:dict = {}
    
    is_initialized:bool = False
    tick_id:int = 0
    accum:float = 0
    N:int = 0
    max_speed:float|None = None
    curr_tick_at:float = 0
    next_tick_at:float = 0
    dt_step:float = config.DEFAULT_DT_BASE
    
    cfg:SimConfig = SimConfig()
    gpu_status:GPUStatus|None
    
    flags:NDArray[np.uint32]
    position:NDArray[np.complex64]
    velocity:NDArray[np.complex64]
    mass:NDArray[np.float32]
    radius:NDArray[np.float32]
    force:NDArray[np.complex64]
    #cgroup:NDArray[np.uint32]
    
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clock = SimClock()
        self._prev = None
        self._uninitialized_bodies:list[message.ParticleInit] = []

    def connect(self, host=SERVER_IPC_HOST, port=SERVER_IPC_PORT):
        self.s.connect((host, port))
        logger.info("Connected to %s %s", host, port)
    
    #def disconnect(self):
    #    return self.send_message(message.MessageType.DISCONNECT)

    def get_valid_indices(self) -> NDArray:
        if not self.is_initialized:
            return np.empty(0, dtype=np.uint32)
        return np.where((self.flags & flags.REMOVED) != flags.REMOVED)[0]

    def find_bodies_at(self, x:float, y:float, margin:float=10):
        if not self.is_initialized:
            return np.empty(0, dtype=np.uint32)
        
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
            logger.warning("Cannot add_particle after initialization.")
        else:
            self._uninitialized_bodies.append(message.ParticleInit(
                flags=flags,
                position=(position.real, position.imag),
                velocity=(velocity.real, velocity.imag),
                mass=mass,
                radius=radius
            ))
        return len(self._uninitialized_bodies) - 1

    def get_particle(self, particle_id:int) -> Particle:
        return ParticleCL(particle_id, self)

    def __iter__(self):
        for i in self.get_valid_indices():
            yield self.get_particle(int(i))
        
    def _apply_config(self, config: SimConfig):
        self.config = config
        logger.info("Config applied: %s", config)

    def _apply_status(self, status: message.StatusResponse):
        self.is_initialized = status.initialized
        self.tick_id = status.tick_id
        self.accum = status.accum
        self.N = status.N
        self.max_speed = status.max_speed
        self.curr_tick_at = status.curr_tick_at
        self.next_tick_at = status.next_tick_at
        self.gpu_status = status.gpu_status
        self.clock.update(status.clock)

    def _connect_shared_memory(self, shared: message.SharedMemoryResponse):
        for f in fields(message.SharedMemoryResponse):
            shm_state = cast(message.SharedMemoryInfo, getattr(shared, f.name))
            shm = shared_memory.SharedMemory(name=shm_state.name, size=shm_state.size, track=False)            
            self._shared[f.name] = shm
            setattr(self, f.name, np.ndarray(shm_state.shape, dtype=shm_state.dtype, buffer=shm.buf))
            logger.info("Connected memory %s %s %s %s %s", f.name, shm_state.name, shm_state.size, shm_state.dtype, shm_state.shape)

    def sync_full_state(self):
        return self.send_message(transport.MessageType.STATE_REQ)

    def send_message(self, message_enum: transport.MessageType, message:Any|None=None) -> bool:
        if message_enum != transport.MessageType.SYNC_REQ:
            logger.info("SEND %s", message_enum.name)
        transport.send_message(self.s, message_enum, message)
        return self._handle_response()

    def _handle_response(self) -> bool:
        _, message_type, payload = transport.recv_message(self.s, transport.MessageType)
        
        if message_type == transport.MessageType.SUCCESS:
            return True
        
        elif message_type == transport.MessageType.INIT_RESP:
            req = message.InitResponse.from_dict(payload)
            self.is_initialized = req.initialized
            self._apply_config(req.config)
            self._connect_shared_memory(req.memory)
            return self.is_initialized
        
        elif message_type == transport.MessageType.STATE_RESP:
            req = message.StateResponse.from_dict(payload)
            self._apply_status(req.status)
            if req.config:
                self._apply_config(req.config)
            if req.memory:
                self._connect_shared_memory(req.memory)
            return True
        
        elif message_type == transport.MessageType.STATUS_RESP:
            req = message.StatusResponse.from_dict(payload)
            self._apply_status(req)
            return True
    
        elif message_type == transport.MessageType.ERROR:
            req = message.ErrorResponse(**payload)
            logger.error(req.error_message)
            # TODO: Emit the error so it can be displayed by the interface
            return False
        
        return False

    def set_device(self, platform_id:int, device_id:int):
        self.device = message.Device(platform_id, device_id)

    def init_sim(self):
        logger.info("Initializing sim with %s particles...", len(self._uninitialized_bodies))
        result = self.send_message(
            message.MessageType.INIT_REQ,
            message.InitRequest(
                particles=self._uninitialized_bodies,
                device=self.device
            ))
        logger.info("Initializing sim completed with result: %s", result)
        return result
    
    def set_clock_speed(self, speed):
        if not self.is_initialized:
            return False
        return self.send_message(message.MessageType.CLOCK_UPDATE, message.ClockUpdateRequest(speed=speed))
    
    def start(self):
        if not self.is_initialized or self.clock.running:
            return False
        return self.send_message(message.MessageType.CLOCK_UPDATE, message.ClockUpdateRequest(running=True))
    
    def stop(self):
        if not self.is_initialized or not self.clock.running:
            return False
        return self.send_message(message.MessageType.CLOCK_UPDATE, message.ClockUpdateRequest(running=False))
    
    def sync(self):
        if not self.is_initialized:
            return False
        return self.send_message(message.MessageType.SYNC_REQ)

    def rel_move(self, ids:Sequence[int], offset:complex):
        if not self.is_initialized:
            return False
        self.send_message(
            message.MessageType.SHIFT_VECTOR_REQ,
            message.ShiftVectorsRequest(
                vector_name='position',
                ids=ids,
                op="add",
                offset=(offset.real, offset.imag)
            )
        )

    def rel_velocity(self, ids:Sequence[int], offset:complex):
        if not self.is_initialized:
            return False
        self.send_message(
            message.MessageType.SHIFT_VECTOR_REQ,
            message.ShiftVectorsRequest(
                vector_name='velocity',
                ids=ids,
                op="mul",
                offset=(offset.real, offset.imag)
            )
        )

    def rel_mass(self, ids:Sequence[int], offset:float):
        if not self.is_initialized:
            return False
        self.send_message(
            message.MessageType.SHIFT_VECTOR_REQ,
            message.ShiftVectorsRequest(
                vector_name='mass',
                ids=ids,
                op="mul",
                offset=(offset.real, 0)
            )
        )

    def tick_once(self):
        self.send_message(message.MessageType.TICK_ONCE)
        self.sync()

    def substep_once(self):
        self.send_message(message.MessageType.SUBSTEP_ONCE)
        self.sync()

    def to_dict(self) -> dict:
        self.sync()
        return {
            "tick_id": int(self.tick_id),
            "N": int(self.N),
            "cfg": asdict(self.cfg),
            
            # particle field vectors
            "flags":    [int(fl) for fl in self.flags],
            "position": [(f"{p.real:.6f}", f"{p.imag:.6f}") for p in self.position],
            "velocity": [(f"{v.real:.6f}", f"{v .imag:.6f}") for v in self.velocity],
            "mass":     [f"{m:.6f}" for m in self.mass],
            "radius":   [f"{r:.6f}" for r in self.radius],
            # force is omitted due to size and lower importance
        }
