import atexit
from threading import BrokenBarrierError
from threading import Event, Lock

import time
import multiprocessing as mp
from multiprocessing import Barrier
from typing import Sequence

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalnp.memory import INTERACTION_MERGED, STATUS_DELETED, STATUS_NOMINAL #, OrbitalMemory
from orbitalengineer.engine.orbitalnp import orbitalmemory, worker
from orbitalengineer.engine.particle import Particle

import numpy as np

from orbitalengineer.engine.simcontroller import CollisionHandler_T, MergeHandler_T, OrbitalSimController, StatusHandler_T

MAX_NUM_PROCESSES = mp.cpu_count() - 1

MIN_BODIES_PER_PROCESS = 50
DEFAULT_SPEED = 1.0
DEFAULT_DT_BASE = 1/20.0
MAX_STEPS_PER_FRAME = 20


class SimController_NP(OrbitalSimController):
    
    bump_buffer:bool = False
    
    strict:bool = False
    bump_blocking:bool = False
    
    _status_handlers:list[StatusHandler_T]
    _merge_handlers:list[MergeHandler_T]
    _collision_handlers:list[CollisionHandler_T]
    
    def __init__(self):
        self.lock = Lock()
        self.bumper = Event() 
        
        self._merge_handlers = []
        self._collision_handlers = []
        self._status_handlers = []
        
        self.current_state:np.typing.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.current_interactions:np.typing.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        
        self._particles:list[Particle] = []
        self.inout_queue = []
        
        self.speed = DEFAULT_SPEED       # user slider: 0.1x … 100x
        self.dt_base = DEFAULT_DT_BASE  # physics step (sec of sim time)
        self.accum = 0.0
        self.last_now:float|None = None
        
        self.num_processes = MAX_NUM_PROCESSES
        self.workers_initialized = False
        self.workers_ready = False
        self.sim_initialized = False

    def add_particle(self, particle:Particle) -> int:
        if self.sim_initialized:
            logger.warning("Cannot add a particle once the simulation has started.")
            return -1
        self._particles.append(particle)
        return len(self._particles) - 1
    
    def find_bodies_at(self, x:float, y:float, margin:float=10):
        indices = self.get_valid_indices()
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.shm.radius[indices] + margin

        # Relative difference between the click and every location
        d = np.complex128(x, y) - self.shm.position[indices]
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius)
        
        # Get the indices 
        return indices[mask]

    def __iter__(self):
        if not self.workers_ready or not self.sim_initialized:
            logger.warning("Cannot iterate over bodies prior to workers being ready.")
            raise SystemExit
            return
        for i in self.get_valid_indices():
            if self.shm.status[i] == STATUS_DELETED:
                continue
            yield self.shm.get_particle(int(i))

    def get_valid_indices(self):
        if hasattr(self, 'ix') and self.ix.size == 0:
            self.ix = np.arange(self.shm.N)
        self.ix = self.ix[self.shm.status[self.ix] == STATUS_NOMINAL]
        return self.ix

    def _init_worker_processes(self):
        init_start = time.perf_counter()
        N = len(self._particles)
        if N < MAX_NUM_PROCESSES * MIN_BODIES_PER_PROCESS:
            self.num_processes = max(1, len(self._particles) // MIN_BODIES_PER_PROCESS)
            logger.info("Number of processes reduced to %s", self.num_processes)
        
        # Used for syncing the worker loop with the tick loop
        barrier_loop = Barrier(self.num_processes)
        
        # Only used for workers, so they can sync their KDK steps.
        self.barrier_sync = Barrier(self.num_processes+1)
        
        logger.info("Starting up %s workers.", self.num_processes)
        for i in range(self.num_processes):
            p = worker.OrbitalWorker(self.shm.N, self.shm.name, barrier_loop, self.barrier_sync, i, self.num_processes)
            p.start()
            atexit.register(lambda: p.kill() if p is not None else None)
    
        logger.info("Workers created. Waiting for them to be ready.")
        self.barrier_sync.wait()
        self.workers_ready = True
        elapsed = time.perf_counter() - init_start
        logger.info("%s workers ready after %.2f ms", self.num_processes, elapsed*1000.0)

    def init_sim(self):
        if self.sim_initialized:
            logger.warning("Attempted to init_sim after simulation was already started.")
            return
        
        
        self._init_buffer()        
        self._init_worker_processes()
        self.sim_initialized = True
        logger.info("Sim was initialized.")
    
    def _init_buffer(self):
        # Initialize the memory and populate the arrays
        self.shm = orbitalmemory.OrbitalMemory(len(self._particles))
        self.ix = np.arange(self.shm.N)
        position = self.shm.position
        velocity = self.shm.velocity
        mass = self.shm.mass
        radius = self.shm.radius
        status = self.shm.status        
        for i, b in enumerate(self._particles):
            position[i] = b.position
            velocity[i] = b.velocity
            mass[i] = b.mass
            radius[i] = b.radius
            status[i] = STATUS_NOMINAL

    def _check_state(self):
        return
        status = self.shm.status
        if self.current_state.size == 0:
            self.current_state = np.copy(status)
            return
        diff = np.argwhere(self.current_state != status).flatten()
        for i in diff:
            for handler in self._status_handlers:
                handler(int(i), self.current_state[i], status[i])
        self.current_state = np.copy(status)
    
    def handle_interaction_changes(self):
        return
        
        interactions = self.shm.interaction[self.buffer_target,]
        if self.current_interactions.size == 0:
            self.current_interactions = np.copy(interactions)
            return
        diff = np.argwhere(self.current_interactions != interactions)
        if not diff.size:
            return
        for i,j in diff:
            self.handle_interaction(
                int(i), int(j),
                int(self.current_interactions[i,j]),
                interactions[i,j]
            )
        self.current_interactions = np.copy(interactions)
    
    def handle_interaction(self, body_i: int, body_j: int, prev: int, next: int):
        if next == INTERACTION_MERGED:
            for handler in self._merge_handlers:
                handler(body_i, body_j)
        # elif next == INTERACTION_COLLIDING:
        #     for handler in self._collision_handlers:
        #         handler(body_i, body_j)

    def add_merge_changed_handler(self, handler:MergeHandler_T):
        self._merge_handlers.append(handler)

    def add_collision_changed_handler(self, handler:MergeHandler_T):
        self._collision_handlers.append(handler)

    def add_status_changed_handler(self, handler:StatusHandler_T):
        self._status_handlers.append(handler)

    def tick(self, now):
        if not self.sim_initialized:
           logger.warning("tick() was called before simulation initialization.")
           return 0
        
        elif not self.workers_ready:
           logger.warning("tick() was called prior to the workers being ready.")
           self._init_worker_processes()
 
        if self.last_now is None:
            self.accum = 0
            self.last_now = now - self.dt_base
 
        wall_dt = now - self.last_now
        self.accum += wall_dt * self.speed

        dt_step = self.dt_base # <- for now, fixed cap

        num_steps = min(self.accum // dt_step, MAX_STEPS_PER_FRAME)
        self.last_now = now
            
        if num_steps > 0:
            self.shm.buffer_ids[0] = np.int64(0)
            self.shm.buffer_ids[1] = np.int64(0)
            # self.shm.buffer_ids[0] = np.int64(self.buffer_source)
            # self.shm.buffer_ids[1] = np.int64(self.buffer_target)
            self.shm.num_steps[:] = np.int64(num_steps)
            self.shm.dt[:] = np.float64(dt_step)
            
            try:
                self.barrier_sync.wait(timeout=10)
                self.barrier_sync.wait(timeout=10)
            except BrokenBarrierError:
                self.workers_ready = False
                logger.warning("Barriers broke. Will attempt to re-init on next tick.")
            
            if self.strict:
                self.accum -= (num_steps * dt_step)
            else:
                self.accum = 0
            
            while len(self.inout_queue) > 0:
                idx, ax, ay = self.inout_queue.pop(0)
                v_curr = self.shm.velocity[idx]
                v_next = v_curr + np.complex128(ax, ay)
                self.shm.velocity[idx] = v_next
                #print(f"{v_curr=:.6f}   {v_next=:.6f}")
        
            
            # if self.bump_buffer:
            #     self.bump_buffer = False
            #     self.buffer_static, self.buffer_target = self.buffer_target, self.buffer_static
            #     self.bumper.set()
            # else:
            #     self.buffer_target = self.buffer_target, self.buffer_source

            self.handle_interaction_changes()
        
        return num_steps
        
    # def bump_buffer_threadsafe(self):
    #     if not self.bump_blocking:
    #         self.buffer_static = self.buffer_source
    #     else:
    #         self.bump_buffer = True
    #         self.bumper.wait()
    #         self.bumper.clear()