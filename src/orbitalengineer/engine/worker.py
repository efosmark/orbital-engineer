import multiprocessing as mp
from multiprocessing.synchronize import Barrier as Barrier_T
import os
import numpy as np
from numba import njit
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.engine.mask import create_pair_partition_mask
from orbitalengineer.engine.memory import INTERACTION_COLLISION, INTERACTION_COLLIDING, INTERACTION_NONE, HISTORY_DEPTH, STATUS_DELETED, STATUS_DELETING, STATUS_NOMINAL, OrbitalMemory, offset


@njit
def worker_filter_jit(
    body_ids,
    status,
    velocity,
    mass,
    radius,
    
):
    ## TODO can this be removed and integrated into kick()
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        if status[i] != STATUS_DELETING:
            continue
        status[i] = STATUS_DELETED
        velocity[i] = 0
        mass[i] = 0
        radius[i] = 0


@njit
def worker_drift_jit(body_ids, r_in, v_in, full_step, r_out):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        r_out[i] = r_in[i] + (v_in[i] * full_step)

@njit
def worker_kick_jit(
    body_ids: NDArray[np.int64],
    half_step: np.float64,
    velocity: NDArray[np.complex128],
    mass: NDArray[np.float64],
    position: NDArray[np.complex128],
    radius: NDArray[np.float64],
    status: NDArray[np.int64],
):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        if status[i] == STATUS_DELETED:
            continue
        for j in np.arange(mass.size):
            if i == j or status[j] == STATUS_DELETED:# or interaction[i,j] != INTERACTION_NONE:
                continue
            
            d = position[j] - position[i]
            dist = np.abs(d) 
            if not dist or not mass[i]:
                continue

            # Detect collisions
            if dist < (radius[i] + radius[j]):
                #interaction[i,j] = INTERACTION_COLLIDING
                if i == j or status[j] == STATUS_DELETED:
                    continue
                
                if mass[i] >= mass[j]:
                    #print(f"colliding {i} and {j}")
                    #interaction[i,j] = INTERACTION_COLLISION
                    combined_mass = mass[i] + mass[j]
                    
                    # Combine velocity by center-of-mass momentum
                    momentum_i = velocity[i] * mass[i]
                    momentum_j = velocity[j] * mass[j]
                    
                    velocity[i] = ((momentum_i + momentum_j) / combined_mass)
                    mass[i] = combined_mass
                    radius[i] = np.sqrt(combined_mass / np.pi) 
                    # todo: shift body to barycenter               
                    
                    # This change does not propagate past the step (unless j is managed by this worker), but is
                    # useful for reducing other collisions 
                    status[j] = STATUS_DELETING
                    
                else:
                    # Flag it to have its properties cleared in the next step
                    #interaction[i,j] = INTERACTION_COLLISION
                    status[i] = STATUS_DELETING
                    mass[i] = 0
                    radius[i] = 0
                    velocity[i] = 0
                continue
        
            # https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
            F = 1.0 * ((mass[i] * mass[j]) / (dist**3)) * d
            acceleration = F / (mass[i])
            velocity[i] += acceleration * half_step


def precompile_njit():
    """ Execute all of the integrator functions, in order to signal numba to compile them.

    Compiling prior to spinning up the workers allows us to avoid having each worker
    compile their own. Instead, they will use the ones handled by the parent process.    
    """
    dt = 0.1
    body_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    status = np.array([0, 0, 0, 0], dtype=np.int64)
    position = np.array([10, 20, 30, 40], dtype=np.complex128)
    velocity = np.array([0, 0, 0, 0], dtype=np.complex128)
    mass = np.array([1, 1, 1, 1], dtype=np.float64)
    radius = np.array([1, 1, 1, 1], dtype=np.float64)
    
    interaction = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int64)

    #handle_collision(np.int64(0), np.int64(1), velocity, mass, interaction, velocity, mass, radius, status)
    worker_kick_jit(body_ids, np.float64(dt/2), velocity, mass, position, radius, status)
    worker_drift_jit(body_ids, radius, velocity, dt, position)
    worker_filter_jit(body_ids, status, velocity, mass, radius)



class OrbitalWorker(mp.Process):
    
    def __init__(self, N:int, name:str, barrier:Barrier_T, barrier_sync:Barrier_T, process_id:int, num_processes:int):
        self.N = N
        self.barrier = barrier
        self.shared_memory_name = name
        self.barrier_sync = barrier_sync
        self.process_id = process_id
        self.num_processes = num_processes
        self.cpu_affinity = (self.process_id % (mp.cpu_count() - 1)) + 1
        super().__init__(daemon=True)
    
    def sync(self):
        self.barrier.wait(timeout=1)

    def run(self):
        os.sched_setaffinity(0, {self.cpu_affinity})

        self.shm = shm = OrbitalMemory(self.N, self.shared_memory_name)
        
        mask = create_pair_partition_mask(shm.N, self.process_id, self.num_processes)
        self.body_ids = np.arange(shm.N, dtype=np.int64)[mask]
        
        logger.info("Worker %s ready, targeting CPU %s.", self.process_id, self.cpu_affinity)
        self.barrier_sync.wait()
        
        while True:
            # Wait for controlling process to set the step_id
            self.barrier_sync.wait(timeout=60)
            
            # Read the current step_id from shared memory
            step_id = np.int64(shm.step[0])
            
            # Read how many steps we should process
            num_steps = np.int64(shm.step[1])
            if num_steps <= 0:
                logger.warning("No step count for step_id=%", step_id)
                continue
            
            dt_full_step = np.float64(shm.step[2])
            for _ in range(num_steps):
                self.step(step_id, dt_full_step)
                step_id += 1
            
            self.barrier_sync.wait()

    def kick(self, step_id:np.int64, dt_half_step:np.float64):
        step_offset = step_id % HISTORY_DEPTH

        # Buffer the current state
        self.sync()
        next_velocity = np.copy(self.shm.velocity[step_offset,])
        next_mass = np.copy(self.shm.mass[step_offset,])
        next_position = np.copy(self.shm.position[step_offset,])
        next_radius = np.copy(self.shm.radius[step_offset,])
        next_status = np.copy(self.shm.status[step_offset,])
        self.sync()
        
        worker_kick_jit(
            self.body_ids,
            dt_half_step,
            next_velocity,
            next_mass,
            next_position,
            next_radius,
            next_status
        )
        
        # Populate the values managed by the worker into shared memory
        self.sync()
        self.shm.velocity[step_offset, self.body_ids] = next_velocity[self.body_ids]
        self.shm.position[step_offset, self.body_ids] = next_position[self.body_ids]
        self.shm.mass[step_offset, self.body_ids] = next_mass[self.body_ids]
        self.shm.radius[step_offset, self.body_ids] = next_radius[self.body_ids]
        self.shm.status[step_offset, self.body_ids] = next_status[self.body_ids]
        self.sync()


    def drift(self, step_id:np.int64, dt_full_step:np.float64):
        step_offset = step_id % HISTORY_DEPTH
        
        self.sync()
        next_position = np.copy(self.shm.position[step_offset,])
        next_velocity = np.copy(self.shm.velocity[step_offset,])
        #interaction = np.copy(self.shm.interaction[self.body_ids,])
        self.sync()
        
        worker_drift_jit(
            self.body_ids,
            next_position,
            next_velocity,
            dt_full_step,
            next_position
        )
        
        self.sync()
        self.shm.velocity[step_offset, self.body_ids] = next_velocity[self.body_ids]
        self.shm.position[step_offset, self.body_ids] = next_position[self.body_ids]
        #self.shm.interaction[self.body_ids,] = interaction[self.body_ids]
        self.sync()

    def step(self, step_id:np.int64, dt_full_step:np.float64):
        half_step = np.float64(dt_full_step / 2.0)
        prev_step_offset = (step_id - 1) % HISTORY_DEPTH
        step_offset = step_id % HISTORY_DEPTH

        # Pre-populate the next step with the former step values
        self.sync()
        self.shm.velocity[step_offset, self.body_ids] = self.shm.velocity[prev_step_offset, self.body_ids]
        self.shm.position[step_offset, self.body_ids] = self.shm.position[prev_step_offset, self.body_ids]
        self.shm.mass[step_offset, self.body_ids] = self.shm.mass[prev_step_offset, self.body_ids]
        self.shm.radius[step_offset, self.body_ids] = self.shm.radius[prev_step_offset, self.body_ids]
        self.shm.status[step_offset, self.body_ids] = self.shm.status[prev_step_offset, self.body_ids]
        self.sync()

        self.kick(step_id, half_step)
        self.drift(step_id, dt_full_step)
        self.kick(step_id, half_step)

        self.sync()
        worker_filter_jit(
            self.body_ids,
            self.shm.status[step_offset,],
            self.shm.velocity[step_offset,],
            self.shm.mass[step_offset,],
            self.shm.radius[step_offset,]
        )
        self.sync()