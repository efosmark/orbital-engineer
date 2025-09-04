import atexit
from collections import deque
import statistics
import time
import multiprocessing as mp
from multiprocessing import Barrier
from multiprocessing.synchronize import Barrier as Barrier_T, Event as Event_T
from typing import Callable

import numpy as np

np.set_printoptions(
    precision=2, 
    edgeitems=1,
    linewidth=30,
    suppress=True,
    threshold=30,
    floatmode="fixed"
)

from orbitalengineer.engine import logger, worker
from orbitalengineer.engine.kernel import integrator_leapfrog_kdk_njit
from orbitalengineer.engine.body import OrbitalBody
from orbitalengineer.engine.memory import HISTORY_DEPTH, INTERACTION_NONE, STATUS_DELETED, STATUS_NOMINAL, BodyProxy, OrbitalMemory, offset

USE_WORKERS = True
MAX_NUM_PROCESSES = mp.cpu_count() - 1
MIN_BODIES_PER_PROCESS = 100

def _default_interaction_handler(i:int, j:int, prev:int, curr:int):
    logger.info("[Interaction] i=%s j=%s prev=%s curr=%s", i, j, prev, curr)

class OrbitalSimController:
    _on_interaction: Callable[[int,int,int,int], bool|None]
    
    def __init__(self):
        self._on_interaction = _default_interaction_handler
        self._bodies:list[OrbitalBody] = []
        self.inout_queue = []
        self.step_id = 1
        
        self.speed = 1.0     # user slider: 0.1x … 100x
        self.dt_base = 1/10.0  # physics step (sec of sim time)
        self.accum = 0.0
        
        self.use_workers = USE_WORKERS
        self.num_processes = MAX_NUM_PROCESSES
        self.workers_initialized = False
        self.workers_ready = False
        self.sim_initialized = False

        self.t_step = deque(maxlen=10)

    def set_interaction_callback(self, cb: Callable[[int,int,int,int], bool|None]):
        self.cb = cb

    def add_body(self, x, y, mass, vx, vy, radius):
        idx = len(self._bodies)
        body = OrbitalBody(x, y, mass, vx, vy, radius, id=idx)
        self._bodies.append(body)
        return idx
    
    def find_bodies_at(self, x:float, y:float, margin:float=10):
        indices = np.arange(self.shm.N)
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.shm.radius[offset(self.step_id), indices] + margin

        # Relative difference between the click and every location
        d = np.complex128(x, y) - self.shm.position[np.int64(self.step_id-1) % HISTORY_DEPTH, indices]
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius) & self.shm.status[offset(self.step_id), indices] == STATUS_NOMINAL
        
        # Get the indices 
        return indices[mask]

    def body(self, idx:int) -> BodyProxy:
        """Get a proxy view for a body at a specific index

        Args:
            idx (int): The index of the body in memory

        Returns:
            BodyProxy: Object allowing for easy retrieval of body properties.
        """
        return BodyProxy(idx, self.shm)

    def __iter__(self):
        if self.use_workers and not self.workers_ready or not self.sim_initialized:
            logger.warning("Cannot iterate over bodies prior to workers being ready.")
            return
        step_offset = offset(self.step_id)
        for i in range(self.shm.N):
            if self.shm.status[step_offset, i] == STATUS_DELETED:
                continue
            yield self.body(int(i))


    def _init_worker_processes(self):
        init_start = time.perf_counter()
        
        # If we precompile in the parent process, the worker processes shouldnt need to.
        worker.precompile_njit()
        logger.info("Precompiling the kernel took %.2f ms.", (time.perf_counter() - init_start) * 1000.0)
        
        # Used for syncing the worker loop with the tick loop
        barrier_loop = Barrier(self.num_processes)
        
        # Only used for workers, so they can sync their KDK steps.
        self.barrier_sync = Barrier(self.num_processes+1)
        
        logger.info("Starting up %s workers.", self.num_processes)
        for i in range(self.num_processes):
            p = worker.OrbitalWorker(self.shm.N, self.shm._shm.name, barrier_loop, self.barrier_sync, i, self.num_processes)
            p.start()
            atexit.register(lambda: p.kill())
    
        logger.info("Workers created. Waiting for them to be ready.")
        self.barrier_sync.wait()
        self.workers_ready = True
        elapsed = time.perf_counter() - init_start
        logger.info("%s workers ready after %.2f ms", self.num_processes, elapsed*1000.0)

    def _tune_strategy(self):
        N = len(self._bodies)
        # No tuning necessary
        if not self.use_workers:
            if N >= MIN_BODIES_PER_PROCESS:
                logger.warning("Simulating %s bodies sequentially. Consider using workers.", N)
            return
        elif N >= MAX_NUM_PROCESSES * MIN_BODIES_PER_PROCESS:
            return

        self.num_processes = max(1, len(self._bodies) // MIN_BODIES_PER_PROCESS)
        
        logger.info("Number of processes reduced to %s", self.num_processes)
        


    def init_sim(self):
        self.last_now = time.perf_counter()

        # Initialize the memory and populate the arrays
        self.shm = OrbitalMemory(len(self._bodies))
        
        self.shm.interaction[:] = INTERACTION_NONE
        self.shm.status[0,] = STATUS_NOMINAL
        for i, b in enumerate(self._bodies):
            self.shm.position[0, i] = np.complex128(b.x, b.y)
            self.shm.velocity[0, i] = np.complex128(b.vx, b.vy)
            self.shm.mass[0, i] = b.mass
            self.shm.radius[0, i] = b.radius
        logger.info("Arrays initialized for %s bodies.", len(self._bodies))
        
        #self._tune_strategy()
        if self.use_workers:
            self._init_worker_processes()
        
        # Full index pairs for use with single-process integrator
        #self.ix, self.jx = np.triu_indices(self.shm.N, k=1)
        #self.mask = (self.shm.status[self.ix] != STATUS_DELETED) & (self.shm.status[self.jx] != STATUS_DELETED)
        self.current_state = np.copy(self.shm.interaction)
        self.sim_initialized = True
        logger.info("Sim was initialized.")
    
    
    def frame(self, now, tick_id):
        if not self.sim_initialized:
           logger.warning("frame() was called before simulation initialization.")
           return 0
        
        if self.use_workers and not self.workers_ready:
           logger.warning("frame() was called prior to the workers being ready.")
           return 0

        wall_dt = now - self.last_now
        self.accum += wall_dt * self.speed

        dt_step = self.dt_base # <- for now, fixed cap
        MAX_STEPS_PER_FRAME = 45

        num_steps = min(self.accum // dt_step, MAX_STEPS_PER_FRAME)
        if num_steps > 0:
            start = time.perf_counter()
            self.shm.step[0] = np.float64(self.step_id)
            self.shm.step[1] = np.float64(num_steps)
            self.shm.step[2] = np.float64(dt_step)
                        
            self.barrier_sync.wait(timeout=10)
            self.barrier_sync.wait()
                    
            self.accum -= (num_steps * dt_step)
            self.step_id += int(num_steps)
            self.last_now = now
            self.t_step.append((time.perf_counter() - start)/num_steps)

        return num_steps

    def _check_state(self):
        mask = self.current_state != self.shm.interaction
        coords = np.argwhere(mask)
        handled = set()
        if coords.size > 0:
            for i, j in coords:
                if (j, i) in handled:
                    continue
                prev = int(self.current_state[i,j])
                curr = int(self.shm.interaction[i,j])
                self._on_interaction(i, j, prev, curr)
                handled.add((i, j))
            self.current_state = np.copy(self.shm.interaction)
