import os
import multiprocessing as mp
from multiprocessing.synchronize import Barrier as Barrier_T
from threading import BrokenBarrierError
import numpy as np    

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalnp import integrator, orbitalmemory
from orbitalengineer.engine.orbitalnp.history import compute_history
from orbitalengineer.engine.orbitalnp.integrator.collision import find_collisions
from orbitalengineer.engine.orbitalnp.integrator.conf import MERGE_STRATEGY_COMBINE
from orbitalengineer.engine.orbitalnp.memory import OrbitalMemory
 
from numba import njit, void, float64, complex128, int64
from numpy.typing import NDArray


@njit(void(
    int64[:],       # body_ids
    complex128[:],  # velocity
    complex128[:],  # velocity
    complex128[:],  # velocity
    complex128[:],  # position
    float64[:],     # radius
    float64[:],     # mass
    float64[:],     # radius
    float64[:],     # mass
    int64[:],       # status
    int64[:],       # status
    # int64[:,:],     # interaction
    # int64[:,:],     # interaction
    # int64[:],       # group
    # int64[:],       # group
    float64[:,:],   # distance
    float64[:,:],   # distance
), cache=True, fastmath=True)
def copy_buffer(
    body_ids: NDArray[np.int64],
    velocity_in: NDArray[np.complex128],
    velocity_out: NDArray[np.complex128],
    position_in: NDArray[np.complex128],
    position_out: NDArray[np.complex128],
    radius_in: NDArray[np.float64],
    radius_out: NDArray[np.float64],
    mass_in: NDArray[np.float64],
    mass_out: NDArray[np.float64],
    status_in: NDArray[np.int64],
    status_out: NDArray[np.int64],
    # interaction_in: NDArray[np.int64],
    # interaction_out: NDArray[np.int64],
    # group_in: NDArray[np.int64],
    # group_out: NDArray[np.int64],
    distance_in: NDArray[np.float64],
    distance_out: NDArray[np.float64],
):
    for i in body_ids:
        velocity_out[i] = velocity_in[i]
        position_out[i] = position_in[i]
        radius_out[i] = radius_in[i]
        mass_out[i] = mass_in[i]
        status_out[i] = status_in[i] 
        # interaction_out[i] = interaction_in[i] 
        # group_out[i] = group_in[i] 
        distance_out[i] = distance_in[i] 




# @njit(void(
#     int64[:],       # body_ids
#     float64[:],     # dt
#     int64[:],       # source
#     int64[:],       # target
#     int64[:],       # num_steps
#     complex128[:],  # velocity
#     complex128[:],  # position
#     float64[:],     # mass
#     int64[:],       # status
#     float64[:],     # radius
#     int64[:,:],     # interaction
#     complex128[:],  # history
#     int64[:]        # history_index
# ), cache=True, fastmath=True)
# def multi_step(
#     body_ids: NDArray[np.int64],
#     dt: np.float64,
#     source: np.int64,
#     target: np.int64,
#     num_steps: np.int64,
#     velocity: NDArray[np.complex128],
#     position: NDArray[np.complex128],
#     mass: NDArray[np.float64],
#     status: NDArray[np.int64],
#     radius: NDArray[np.float64],
#     interaction: NDArray[np.int64],
#     history: NDArray[np.complex128],
#     history_index: NDArray[np.int64]
# ):
#     copy_buffer(
#         body_ids,
#         velocity_in=velocity[source],
#         velocity_out=velocity,
#         position_in=position[source],
#         position_out=position,
#         mass_in=mass[source],
#         mass_out=mass,
#         radius_in=radius[source],
#         radius_out=radius,
#         status_in=status[source],
#         status_out=status,
#         interaction=interaction
#     )
    
#     # integrator.kick(
#     #     body_ids,
#     #     dt / 2.0,
#     #     velocity=velocity,
#     #     mass=mass,
#     #     position=position,
#     #     radius=radius,
#     #     status=status,
#     # )
    
#     for i in np.arange(0, num_steps):        
#         integrator.drift(
#             body_ids,
#             dt,
#             status=status,
#             radius=radius,
#             velocity=velocity,
#             position=position,
#             interaction=interaction
#         )
        
#         integrator.merge(
#             body_ids,
#             mass=mass,
#             velocity=velocity,
#             position=position,
#             status=status,
#             radius=radius,
#             interaction=interaction,
#         )
        
#         if i < num_steps - 1:
#             integrator.kick(
#                 body_ids,
#                 dt,
#                 velocity=velocity,
#                 position=position,
#                 mass=mass,
#                 radius=radius,
#                 status=status,
#             )
    
#     integrator.kick(
#         body_ids,
#         dt / 2.0,
#         velocity=velocity,
#         position=position,
#         mass=mass,
#         radius=radius,
#         status=status,
#     )
     
#     compute_history(
#         body_ids,
#         target,
#         position,
#         history,
#         history_index
#     )


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
    
    def sync(self):
        self.barrier.wait(timeout=1)

    def run(self):
        os.sched_setaffinity(0, {self.cpu_affinity})

        self.shm = shm = orbitalmemory.OrbitalMemory(self.N, self.shared_memory_name)
        self.mask = create_pair_partition_mask(shm.N, self.process_id, self.num_processes)
        self.body_ids = np.arange(shm.N, dtype=np.int64)[self.mask]
        
        logger.info("Worker %s ready, targeting CPU %s.", self.process_id, self.cpu_affinity)
        try:
            self.barrier_sync.wait(timeout=1)
        except BrokenBarrierError:
            logger.warning("Worker %s failed initial sync.", self.process_id)
        
        try:
            while True:
                # Wait for controlling process to set the buffer ids and num steps
                self.barrier_sync.wait(timeout=10)
                
                buffer_source, buffer_target = shm.buffer_ids
                
                # Read how many steps we should process
                num_steps = shm.num_steps[0]
                if num_steps <= 0:
                    continue
                
                dt = shm.dt[0]
                self.multi_step(buffer_source, buffer_target, num_steps, dt)

                self.barrier_sync.wait(timeout=10) 
        
        except BrokenBarrierError:
            logger.warning("Worker %s failed to sync with controller. Closing out.", self.process_id)

    def multi_step(self, source, target, num_steps, dt):
        #self.sync()
        
        # multi_step(
        #     body_ids=self.body_ids,
        #     dt=dt,
        #     source=source,
        #     target=target,
        #     num_steps=num_steps,
        #     velocity=self.shm.velocity,
        #     position=self.shm.position,
        #     mass=self.shm.mass,
        #     status=self.shm.status,
        #     radius=self.shm.radius,
        #     interaction=self.shm.interaction,
        #     history=self.shm.history,
        #     history_index=self.shm.history_index
        # )
        
        target = 0
        half_step = np.float32(dt / 2.0)
                
        # copy_buffer(
        #     self.body_ids,
        #     velocity_in=self.shm.velocity[source],
        #     velocity_out=self.shm.velocity,
        #     position_in=self.shm.position[source],
        #     position_out=self.shm.position,
        #     mass_in=self.shm.mass[source],
        #     mass_out=self.shm.mass,
        #     radius_in=self.shm.radius[source],
        #     radius_out=self.shm.radius,
        #     status_in=self.shm.status[source],
        #     status_out=self.shm.status,
        #     #interaction_in=self.shm.interaction[source],
        #     #interaction_out=self.shm.interaction,
        #     #group_in=self.shm.group[source],
        #     #group_out=self.shm.group,
        #     distance_in=self.shm.distance[source],
        #     distance_out=self.shm.distance,
        # )
        
        #self.sync()
        
        #assert self.cpu_affinity in os.sched_getaffinity(0), "Affinity not applied"

        
        integrator.kick(
            self.body_ids,
            half_step,
            velocity=self.shm.velocity,
            position=self.shm.position,
            mass=self.shm.mass,
            radius=self.shm.radius,
        )
        
        #self.sync()
        
        for i in range(num_steps):
            self.sync()
            integrator.drift(
                self.body_ids,
                dt,
                velocity=self.shm.velocity,
                position=self.shm.position,
            )
            
            
            # if MERGE_STRATEGY_COMBINE:
            #     #self.sync()
            #     integrator.merge(
            #         self.body_ids,
            #         distance=self.shm.distance,
            #         mass=self.shm.mass,
            #         velocity=self.shm.velocity,
            #         position=self.shm.position,
            #         status=self.shm.status,
            #         radius=self.shm.radius,
            #     )
            
            self.sync()
            
            if i < num_steps - 1:
                integrator.kick(
                    self.body_ids,
                    dt,
                    velocity=self.shm.velocity,
                    position=self.shm.position,
                    mass=self.shm.mass,
                    radius=self.shm.radius,
                )
        
        #self.sync()
        
        integrator.kick(
            self.body_ids,
            half_step,
            velocity=self.shm.velocity,
            position=self.shm.position,
            mass=self.shm.mass,
            radius=self.shm.radius,
        )
        
        #self.sync()
           
        # compute_history(
        #     self.body_ids,
        #     target,
        #     self.shm.position,
        #     self.shm.history,
        #     self.shm.history_index
        # )
        
        #self.sync()