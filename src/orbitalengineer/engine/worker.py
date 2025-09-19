import multiprocessing as mp
from multiprocessing.synchronize import Barrier as Barrier_T
import os
import numpy as np

from orbitalengineer.engine import logger
from orbitalengineer.engine import integrator
from orbitalengineer.engine.history import compute_history
from orbitalengineer.engine.memory import OrbitalMemory


def create_pair_partition_mask(num_pairs, partition_id, num_partitions):
    indices = np.arange(num_pairs)
    split_indices = np.array_split(indices, num_partitions)
    
    mask = np.zeros(num_pairs, dtype=bool)
    mask[split_indices[partition_id]] = True
    return mask


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
    
    # def get_valid_indices(self, step_id):
    #     mask = self.mask & (self.shm.mass[step_id % 2,] > 0)
    #     self.body_ids = np.arange(self.shm.N, dtype=np.int64)[mask]
    #     return self.body_ids
    
    def sync(self):
        self.barrier.wait(timeout=1)

    def run(self):
        os.sched_setaffinity(0, {self.cpu_affinity})

        self.shm = shm = OrbitalMemory(self.N, self.shared_memory_name)
        self.mask = create_pair_partition_mask(shm.N, self.process_id, self.num_processes)
        self.body_ids = np.arange(shm.N, dtype=np.int64)[self.mask]
        
        logger.info("Worker %s ready, targeting CPU %s.", self.process_id, self.cpu_affinity)
        self.barrier_sync.wait()
        
        while True:
            # Wait for controlling process to set the step_id
            self.barrier_sync.wait(timeout=60)
            
            # Read the current step_id from shared memory
            step_id = shm.step_id[0]
            
            # Read how many steps we should process
            num_steps = shm.num_steps[0]
            if num_steps <= 0:
                logger.warning("No step count for step_id=%", step_id)
                continue
            
            dt = shm.dt[0]
            self.multi_step(step_id, num_steps, dt)
            self.barrier_sync.wait() 


    def kick(self, step_id:np.int64, dt:np.float64):
        self.sync()
        step_offset = step_id % 2
        next_velocity = self.shm.velocity[step_offset,]
        next_mass = self.shm.mass[step_offset,]
        next_position = self.shm.position[step_offset,]
        integrator.kick(
            self.body_ids,
            dt,
            velocity=next_velocity,
            mass=next_mass,
            position=next_position,
            radius=self.shm.radius[step_offset,],
            status=self.shm.status
        )
        self.sync()
    

    def drift(self, step_id:np.int64, dt:np.float64):
        self.sync()
        prev_step_offset = (step_id - 1) % 2
        step_offset = step_id % 2
        integrator.drift(
            self.body_ids,
            dt,
            self.shm.status,
            self.shm.radius[step_offset,],
            self.shm.velocity[step_offset,],
            self.shm.position[prev_step_offset,],
            self.shm.position[step_offset,],
            self.shm.interaction
        )
        self.sync()
    
    def prepopulate(self, step_id):
        prev_step_offset = (step_id - 1) % 2
        step_offset = step_id % 2

        # Pre-populate the next step with the former step values
        self.sync()
        self.shm.velocity[step_offset, self.body_ids] = self.shm.velocity[prev_step_offset, self.body_ids]
        self.shm.position[step_offset, self.body_ids] = self.shm.position[prev_step_offset, self.body_ids]
        self.shm.mass[step_offset, self.body_ids] = self.shm.mass[prev_step_offset, self.body_ids]
        self.shm.radius[step_offset, self.body_ids] = self.shm.radius[prev_step_offset, self.body_ids]
        self.sync()

    def merge(self, step_id):
        step_offset = step_id % 2
        self.sync()
        integrator.merge(
            self.body_ids,
            self.shm.interaction,
            mass=self.shm.mass[step_offset,],
            velocity=self.shm.velocity[step_offset,],
            position=self.shm.position[step_offset,],
            radius=self.shm.radius[step_offset,],
            status=self.shm.status
        )
        self.sync()

    def multi_step(self, step_id, num_steps, dt):
        half_step = np.float64(dt / 2.0)
        self.prepopulate(step_id)
        
        self.kick(step_id, half_step)
        for i in range(num_steps):
            
            self.drift(step_id, dt)
            self.merge(step_id)
            
            if i < num_steps - 1:
                self.kick(step_id, dt)
                step_id += 1
                self.prepopulate(step_id)
        
        self.kick(step_id, half_step)
        
        self.sync()
        compute_history(
            self.body_ids,
            np.int64(step_id),
            self.shm.position,
            self.shm.history,
            self.shm.history_index
        
        )
        self.sync()
        