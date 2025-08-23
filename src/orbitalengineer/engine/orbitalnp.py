import atexit
import time
import multiprocessing as mp
from multiprocessing import Barrier
from multiprocessing.synchronize import Barrier as Barrier_T, Event as Event_T
from typing import Callable

import numpy as np

from orbitalengineer.engine import logger, worker_force_vectors
from orbitalengineer.engine.kernel import integrator_leapfrog_kdk_njit
from orbitalengineer.engine.body import OrbitalBody
from orbitalengineer.engine.memory import BodyProxy, OrbitalMemory
#from scratch import worker_force_vectors

NUM_PROCESSES = 1
SOFTENING_PARAMETER = 5.0

        
# def compute_forces(N, mem:OrbitalMemory, i:int, G=1.0):
#     jx = np.arange(mem.mass.size)
#     ix = np.ndarray(mem.mass.size, dtype=np.int16)
#     ix.fill(i)
    
#     dx = mem.x[jx] - mem.x[ix]
#     dy = mem.y[jx] - mem.y[ix]
    
#     # Apply Newton's Law of Universal Gravitation
#     dist_sq = dx**2 + dy**2
#     inv_dist3 = 1.0 / (dist_sq + SOFTENING_PARAMETER**2)**1.5

#     # Gravitational force magnitude (symmetric)
#     force_mag = G * mem.mass[ix] * mem.mass[jx] * inv_dist3
#     return abs(force_mag)

def _default_interaction_handler(i:int, j:int, prev:int, curr:int):
    logger.debug("[Interaction] i=%s j=%s prev=%s curr=%s", i, j, prev, curr)


class OrbitalSimController:
    
    _on_interaction: Callable[[int,int,int,int], bool|None]
    
    def __init__(self):
        self._on_interaction = _default_interaction_handler
        self._bodies:list[OrbitalBody] = []
        self.inout_queue = []
        
        self.speed = 1.0               # user slider: 0.1x … 100x
        self.dt_base = 1/200           # physics step (sec of sim time)
        self.accum = 0.0
        
        self.workers_initialized = False
        self.workers_ready = False
        self.sim_initialized = False

    def set_interaction_callback(self, cb: Callable[[int,int,int,int], bool|None]):
        self.cb = cb

    def add_body(self, x, y, mass, vx, vy, radius):
        idx = len(self._bodies)
        body = OrbitalBody(x, y, mass, vx, vy, radius, id=idx)
        self._bodies.append(body)
        return idx
    
    def find_bodies_at(self, x:float, y:float, margin:float=10):
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.shm.radius + margin

        # Relative difference between the click and every location
        d = np.complex128(x, y) - self.shm.r
        
        # Create a mask indicating where there is crossover
        mask = np.abs(d) <= radius
        
        # Get the indices 
        return np.arange(self.shm.N)[mask]

    def body(self, idx:int) -> BodyProxy:
        """Get a proxy view for a body at a specific index

        Args:
            idx (int): The index of the body in memory

        Returns:
            BodyProxy: Object allowing for easy retrieval of body properties.
        """
        return BodyProxy(idx, self.shm)

    def __iter__(self):
        if not self.workers_ready:
            return
        for i in range(self.reader.mass.size):
            if self.reader.mass[i] <= 0:
                continue
            yield self.body(i)

    def _init_processes(self):
        init_start = time.perf_counter()
        
        self.barrier_init = Barrier(NUM_PROCESSES+1)
        self.barrier_loop = Barrier(NUM_PROCESSES+1)
        
        NUM_FORCE_VECTOR_WORKERS = 20
        for i in range(NUM_FORCE_VECTOR_WORKERS):
            p = mp.Process(
                target=worker_force_vectors.worker_force_vectors,
                args=(self.shm.N, self.shm._shm.name, self.barrier_init, self.barrier_loop, i, NUM_FORCE_VECTOR_WORKERS),
                daemon=True
            )
            p.start()
            atexit.register(lambda: p.kill())
    
        self.barrier_init.wait()
        self.workers_ready = True
        elapsed = time.perf_counter() - init_start
        logger.debug("%s workers ready after %.2f} ms", NUM_PROCESSES, elapsed*1000.0)


    def init_sim(self):
        self.last_now = time.perf_counter()

        #
        # Initialize the memory and populate the arrays
        #
        self.reader = self.shm = OrbitalMemory(len(self._bodies))
        for i, b in enumerate(self._bodies):
            self.reader.r[i] = self.shm.r[i] = np.complex128(b.x, b.y)
            self.reader.mass[i] = self.shm.mass[i]  = b.mass
            self.reader.v[i] = self.shm.v[i] = np.complex128(b.vx, b.vy)
            self.reader.radius[i] = self.shm.radius[i]  = b.radius
        logger.debug("Arrays initialized.")
        
        #self._init_processes()
        self.workers_ready = True
        
        # Full index pairs for use with single-process integrator
        self.ix, self.jx = np.triu_indices(self.shm.N, k=1)
        
        self.current_state = np.copy(self.shm.state)
        
    def frame(self, now):
        if not self.sim_initialized:
            self.sim_initialized = True
            self.init_sim()
        
        if not self.workers_ready:
            logger.warning("frame() was called prior to the workers being ready.")
            return 0
        
        wall_dt = now - self.last_now
        self.accum += wall_dt * self.speed

        dt_step = self.dt_base            # <- for now, fixed cap
        steps = 0
        MAX_STEPS_PER_FRAME = 60
        
        while self.accum >= dt_step and steps < MAX_STEPS_PER_FRAME:
            
            
            #self.ix = np.argwhere(self.shm.mass[self.ix])
            #self.jx = np.argwhere(self.shm.mass[self.jx])
            mask = (self.shm.mass[self.ix] > 0) & (self.shm.mass[self.jx] > 0)
            self.ix, self.jx = self.ix[mask], self.jx[mask]

            integrator_leapfrog_kdk_njit(
                self.ix,
                self.jx,
                self.shm.v,
                self.shm.r,
                self.shm.a,
                self.shm.radius,
                self.shm.mass,
                self.shm.distance,
                self.shm.state,
                np.float64(dt_step)
            )
            
            self.accum -= dt_step
            steps += 1
        
        self._check_state()
        self.last_now = now
        return steps

    def _check_state(self):
        mask = self.current_state != self.shm.state
        coords = np.argwhere(mask)
        handled = set()
        if coords.size > 0:
            for i, j in coords:
                if (j, i) in handled:
                    continue
                prev = int(self.current_state[i,j])
                curr = int(self.shm.state[i,j])
                self._on_interaction(i, j, prev, curr)
                handled.add((i, j))
            self.current_state = np.copy(self.shm.state)
