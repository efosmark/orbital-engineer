import pyopencl as cl
import numpy as np
from orbitalengineer import flags
from orbitalengineer.engine.config import MAX_NUM_CONTACTS_PER_BODY
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "contacting/contacting.cl"

class FindContactingBodiesPipeline(PipelineComponent):
    debug_flag = "contacting"

    def initialize(self):
        self._find_contacting_bodies = self._load_kernel("find_contacting_bodies", KERNEL_FILE_LOCATION)
        self.n_direct_contacts = self.alloc(self.N, dtype=np.uint32)
        self.direct_contacts = self.alloc(self.N * self.N, dtype=np.uint32)
        
    def get_ids(self, flags_buffer:cl.Buffer):
        n_direct_contacts = self.get_host_vector(self.n_direct_contacts, sync=True)
        flags_host = self.get_host_vector(flags_buffer, sync=True)
                        
        all_colliding_ids = np.array([
            i for i in range(n_direct_contacts.size) 
            if n_direct_contacts[i] > 0 and not (flags_host[i]|flags.REMOVED)
        ], dtype=np.uint32)
        if all_colliding_ids.size == 0:
            return None, 0, 0
        
        local_size = min(int(np.max(n_direct_contacts)) + 1, 64)
        if local_size <= 0 or all_colliding_ids.size == 0:
            return None, 0, 0
        
        global_size = all_colliding_ids.size * local_size
        return self._create_buffer(all_colliding_ids), global_size, local_size
     
    def _print_collisions_per_body(self):
        """Debug printing of the collision matrix."""
        n_direct_contacts = self.get_host_vector(self.n_direct_contacts, sync=True)
        direct_contacts = self.get_host_vector(self.direct_contacts, sync=True)
        for i in range(self.N):
            try:
                if n_direct_contacts[i] <= 1: continue
                print(f" {i:3.0f} [{n_direct_contacts[i]:2.0f}] | ", end="")
                contacts = [
                    f"{direct_contacts[(self.N * i) + j]:3.0f}"
                    for j in range(n_direct_contacts[i])
                ]
                print(" ".join(contacts))
            except IndexError:
                break

    def find_contacting_bodies(self, state:PrimaryStateVectors, distance:DistancePipeline):        
        self._find_contacting_bodies(
            self.queue,
            state.grid_stride_global_size,
            state.grid_stride_local_size,
            
            # Args
            np.uint32(self.N),
            state.flags,
            distance.is_touching,
            self.n_direct_contacts,
            self.direct_contacts
        )

    def __call__(self, state:PrimaryStateVectors, distance:DistancePipeline):
        self.find_contacting_bodies(state, distance)
        #self.find_contacting_bodies_on_host(state, distance)
        return (self.n_direct_contacts, self.direct_contacts)