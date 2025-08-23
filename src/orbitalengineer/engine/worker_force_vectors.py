from multiprocessing.synchronize import Barrier as Barrier_T
import numpy as np

from orbitalengineer.engine.mask import create_pair_partition_mask
from orbitalengineer.engine.memory import OrbitalMemory, fp

def worker_force_vectors(N, name:str, barrier_init:Barrier_T,  barrier_loop:Barrier_T, process_id:int, num_processes:int):
    # Core 0-4 are reserved for GTK & tick loop
    #os.sched_setaffinity(0, {process_id+4})
    shm = OrbitalMemory(N, name)
    
    ix, jx = np.triu_indices(N, k=1)
    sorted_idx = np.argsort(ix * shm.N + jx)
    ix = ix[sorted_idx]
    jx = jx[sorted_idx]
    
    # Select the partition of pairs for this worker
    mask = create_pair_partition_mask(len(ix), process_id, num_processes)
    ix, jx = ix[mask], jx[mask]
    
    barrier_init.wait()
    while True:
        barrier_loop.wait()
        
        #mask = (shm.mass[ix] > 0) & (shm.mass[jx] > 0)
        #ix = ix[mask]
        #jx = jx[mask]
        
        for k in np.arange(ix.size):
            i = ix[k]
            j = jx[k]

            dx = shm.x[j] - shm.x[i]
            dy = shm.y[j] - shm.y[i]
            dist_sq = dx * dx + dy * dy
            inv_dist3 = 1.0 / ((dist_sq + 0.1**2) ** 1.5)

            force_mag = shm.mass[i] * shm.mass[j] * inv_dist3
            fx = fp(dx * force_mag)
            fy = fp(dy * force_mag)

            shm.ax[i] += fp(fx / (shm.mass[i] + 1e-8))
            shm.ay[i] += fp(fy / (shm.mass[i] + 1e-8))

            shm.ax[j] -= fp(fx / (shm.mass[j] + 1e-8))
            shm.ay[j] -= fp(fy / (shm.mass[j] + 1e-8))

        barrier_loop.wait()
