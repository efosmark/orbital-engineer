import atexit
import json
import socket
from multiprocessing import shared_memory
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.particle_cl import ParticleCL
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ipc.response import _SharedMemoryState, ConfigResponse, ServerResponse, SharedMemoryStateResponse

HOST = ''    # The remote host
PORT = 50008 # The same port as used by the server

class ClientSocketConnection:
    _shared:dict = {}
    
    is_initialized:bool = False
    tick_id:int = 0

    accum:float = 0
    dt_base:float
    N:int
    G:float
    coef_of_restitution:float
    EPS_DIST:float
    EPS_TIME:float
    
    flags:NDArray[np.uint32]
    position:NDArray[np.complex64]
    velocity:NDArray[np.complex64]
    mass:NDArray[np.float32]
    radius:NDArray[np.float32]
    force:NDArray[np.complex64]
    
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.clock = SimClock()
        atexit.register(self.disconnect)
        
        self._prev = None
        self._uninitialized_bodies:list[Particle] = []

    def disconnect(self):
        self.s.close()
        for name, shm in self._shared.items():
            try:
                shm.close()
            except FileNotFoundError:
                ...

    def get_valid_indices(self) -> NDArray:
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

    def add_particle(self, particle:Particle):
        if self.is_initialized:
            #logger.warning("Cannot add a particle once the simulation has started.")
            return -1
        self._uninitialized_bodies.append(particle)
        return len(self._uninitialized_bodies) - 1

    def get_particle(self, particle_id:int) -> Particle:
        return ParticleCL(particle_id, self)

    def __iter__(self):
        for i in self.get_valid_indices():
            yield self.get_particle(int(i))

    def send_packet(self, data):
        self.s.sendall(json.dumps(data).encode() + b'\n')
        resp = json.loads(self.s.recv(1024).strip())
        r = ServerResponse.from_dict(resp)
        self.handle_response(r)
        return resp

    def handle_response(self, resp:ServerResponse):
        if resp.status:
            self.accum = resp.status.accum
            self.tick_id = resp.status.tick_id
        if resp.memory:
            self._connect_shared_memory(resp.memory)
        if resp.config:
            self._set_config_defs(resp.config)
        if resp.device:
            self.platform_id = resp.device.platform_id
            self.device_id = resp.device.device_id
        if resp.clock:
            self.clock.duration = resp.clock.duration
            self.clock.speed = resp.clock.speed
            self.clock.running = resp.clock.running
            self.clock.last_time_ms = resp.clock.last_time_ms

    def _connect_shared_memory(self, mem):
        for field in SharedMemoryStateResponse._FIELDS:
            shm_state = cast(_SharedMemoryState, getattr(mem, field))
            shm = shared_memory.SharedMemory(name=shm_state.name, size=shm_state.size)            
            self._shared[field] = shm
            setattr(self, field, np.ndarray(shm_state.shape, dtype=shm_state.dtype, buffer=shm.buf))
    
    def _set_config_defs(self, config:ConfigResponse):
        self.N = config.N
        self.G = config.G
        self.coef_of_restitution = config.coef_of_restitution
        self.dt_base = config.dt_base
        self.EPS_DIST = config.EPS_DIST
        self.EPS_TIME = config.EPS_TIME

    def set_cl_device(self, platform_id:int, device_id:int):
        return self.send_packet({
            'device': {
                'platform_id': platform_id,
                'device_id': device_id
            }
        })

    def load_from_dict(self, data):
        return self.send_packet({
            'load': data["orbital"],
            'clock': data["clock"]
        })

    def init_sim(self):
        init = self.send_packet({
            'init': True,
            'particles': [ p.asdict() for p in self._uninitialized_bodies ]
        })
        return 'init' in init
    
    def set_clock_speed(self, speed):
        return self.send_packet({
            'clock': {
                'speed':speed
            }
        })
    
    def reset(self):
        return self.send_packet({ 'reset': True })
    
    def start(self):
        return self.send_packet({ 'start': True })
    
    def stop(self):
        return self.send_packet({ 'stop': True })
    
    def sync(self):
        return self.send_packet({ 'sync': True })
    
    def tick(self, now):
        return self.send_packet({ 'tick': now })

    def rel_move(self, ids:Sequence[int], offset_x:float, offset_y:float):
        self.send_packet({
            'rel_move': {
                'ids': ids,
                'offset': [offset_x, offset_y]
            }
        })

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
