from multiprocessing.synchronize import Barrier as Barrier_T
import os
import numpy as np
from numba import njit

from orbitalengineer.engine import kernel, logger
from orbitalengineer.engine.mask import create_pair_partition_mask
from orbitalengineer.engine.memory import INTERACTION_COLLIDED, INTERACTION_COLLIDING, INTERACTION_NONE, STATUS_DELETED, STATUS_NOMINAL, OrbitalMemory


@njit
def worker_drift_jit(body_ids, r_in, v_in, full_step, r_out):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        r_out[i] = r_in[i] + (v_in[i] * full_step)


@njit
def worker_kick_jit(body_ids, v_in, mass, position, interaction, radius, half_step, v_out, distance):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        v_out[i] = v_in[i]
        for j in np.arange(mass.size):
            if i == j or interaction[i,j] != INTERACTION_NONE:
                continue
            
            d = position[j] - position[i]
            dist = distance[j,i] = np.abs(d)
            if not dist:
                continue
        
            # Detect collisions
            if dist < (radius[i] + radius[j]):
                interaction[i,j] = interaction[j,i] = INTERACTION_COLLIDING
                continue

            # https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation#Vector_form
            F = 1.0 * ((mass[i] * mass[j]) / (dist**3)) * d
            
            acceleration = F / (mass[i] + 1e-8)
            v_out[i] += acceleration * half_step


@njit
def worker_collide_jit(body_ids, velocity, mass, radius, interaction, distance, status):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        
        for j in np.arange(mass.size):
            if i == j or status[i] != STATUS_NOMINAL or interaction[i,j] != INTERACTION_COLLIDING or mass[i] == 0:
                continue
            
            interaction[i,j] = INTERACTION_COLLIDED
            distance[i,j] = 0
            if mass[i] >= mass[j]:
                combined_mass = mass[i] + mass[j]
                if not combined_mass:
                    continue
                
                # Combine velocity by center-of-mass momentum
                momentum_i = velocity[i] * mass[i]
                momentum_j = velocity[j] * mass[j]
                
                velocity[i] = ((momentum_i + momentum_j) / combined_mass)
                mass[i] = combined_mass
                radius[i] = np.sqrt(mass[i] / np.pi)
            
            else:
                # Flag it to have its properties cleared in the next step
                status[i] = STATUS_DELETED


@njit
def worker_filter_jit(body_ids, velocity, mass, radius, interaction, distance, status):
    for k in np.arange(body_ids.size):
        i = body_ids[k]
        if status[i] != STATUS_DELETED:
            continue
        velocity[i] = 0
        mass[i] = 0
        radius[i] = 0


@njit
def precompile_njit():
    """ Execute all of the integrator functions, in order to signal numba to compile them.

    Compiling prior to spinning up the workers allows us to avoid having each worker
    compile their own. Instead, they will use the ones handled by the parent process.    
    """
    dt = 0.1
    body_ids = np.array([0, 1, 2, 3], dtype=np.uint)
    status = np.array([0, 0, 0, 0], dtype=np.uint)
    position = np.array([10, 20, 30, 40], dtype=np.complex128)
    velocity = np.array([0, 0, 0, 0], dtype=np.complex128)
    mass = np.array([1, 1, 1, 1], dtype=np.float64)
    radius = np.array([1, 1, 1, 1], dtype=np.float64)
    interaction = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint)
    distance = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)

    worker_kick_jit(body_ids, velocity, mass, position, interaction, radius, dt/2, velocity, distance)
    worker_drift_jit(body_ids, radius, velocity, dt, position)
    worker_collide_jit(body_ids, velocity, mass, radius, interaction, distance, status)
    worker_filter_jit(body_ids, velocity, mass, radius, interaction, distance, status)


def worker_integrator_kdk(N, name:str, barrier_loop:Barrier_T, barrier_sync:Barrier_T, process_id:int, num_processes:int):
    # Core 0-4 are reserved for GTK & tick loop
    os.sched_setaffinity(0, {process_id+4})
    
    shm_A = OrbitalMemory(N, name)
    
    mask = create_pair_partition_mask(shm_A.mass.size, process_id, num_processes)
    body_ids = np.arange(shm_A.mass.size)[mask]
    
    logger.info("Worker %s ready.", process_id)
    barrier_sync.wait()
    
    last_step_id = -1
    while True:
        
        # Filter out deleted bodies
        mask = shm_A.status[body_ids] != STATUS_DELETED
        body_ids = body_ids[mask]
        
        # Wait for controlling process to set the step_id
        barrier_sync.wait(timeout=5)
        
        full_step = np.float64(shm_A.step[1])
        half_step = np.float64(full_step / 2.0)

        step_id = shm_A.step[0]
        if step_id == last_step_id:
           logger.warning("Worker %s is spinning too quickly. step=%s   last=%s", process_id, step_id, last_step_id)
        elif step_id != last_step_id + 1:
           logger.warning("Worker %s is behind by %s steps.", process_id, abs(step_id - last_step_id))
        last_step_id = step_id
        
        velocity = shm_A.v
        position = shm_A.r
        mass = shm_A.mass
        interaction = shm_A.interaction
        radius = shm_A.radius
        distance = shm_A.distance
        status = shm_A.status
        
        worker_kick_jit(body_ids, velocity, mass, position, interaction, radius, half_step, velocity, distance)
        barrier_loop.wait(timeout=2)
        
        worker_drift_jit(body_ids, position, velocity, full_step, position)
        barrier_loop.wait(timeout=2)
        
        worker_kick_jit(body_ids, velocity, mass, position, interaction, radius, half_step, velocity, distance)
        barrier_loop.wait(timeout=2)

        worker_collide_jit(body_ids, velocity, mass, radius, interaction, distance, status)
        barrier_loop.wait(timeout=2)

        worker_filter_jit(body_ids, velocity, mass, radius, interaction, distance, status)
        barrier_loop.wait(timeout=2)

        # Sync with parent process loop
        barrier_sync.wait(timeout=5)