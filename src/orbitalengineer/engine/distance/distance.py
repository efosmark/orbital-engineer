import pyopencl as cl
import numpy as np
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

mf = cl.mem_flags

DEFAULT_MEM_FLAGS = mf.READ_WRITE|mf.COPY_HOST_PTR

KERNEL_FILE_LOCATION = "distance/distance.cl"

class DistancePipeline(PipelineComponent):
    debug_flag = 'distance'
    
    _evt_edge_distance:cl.Event|None = None
    
    def initialize(self):
        self._knl_edge_distance = self._load_kernel("edge_distance", KERNEL_FILE_LOCATION)
        self.edge_to_edge = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.float32)
        self.is_touching = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.bool)
        self.is_nearby = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.bool)

        self.n_all_colliding_ids = self.vec.alloc(1, dtype=np.uint32, mem_flags=mf.READ_WRITE|mf.USE_HOST_PTR)
        self.all_colliding_ids = self.vec.alloc(self.vec.N, dtype=np.uint32)
    
        self.n_direct_contacts = self.vec.alloc(self.vec.N, dtype=np.uint32, shared_name="n_direct_contacts")
        self.direct_contacts = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.uint32, shared_name="direct_contacts")

        self.n_nearby_contacts = self.vec.alloc(self.vec.N, dtype=np.uint32)
        self.nearby_contacts = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.uint32)
    
    def get_all_colliding_ids(self):
        n_all_colliding_ids = self.vec.get_host_vector(self.n_all_colliding_ids)
        with self.tr('n_all_colliding_ids (sync)'):
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, n_all_colliding_ids[:1], self.n_all_colliding_ids, wait_for=[self._evt_edge_distance] if self._evt_edge_distance is not None else None)) #.wait()
            n_all_colliding_ids = n_all_colliding_ids[0]
        if n_all_colliding_ids == 0: return None, 0, 0
        local_size = min(n_all_colliding_ids, 64)
        global_size = n_all_colliding_ids * local_size
        return self.all_colliding_ids, global_size, local_size

    def __call__(self, state:PrimaryStateVectors):
        self._evt_edge_distance = self._knl_edge_distance(
                self.queue,
                (self.vec.N_bodies_valid * self.vec.Lx, ),
                (self.vec.Lx, ),
                
                # Args
                np.uint32(self.vec.N_bodies_alloc),
                np.uint32(self.vec.N_bodies_valid),
                state.flags,
                state.position,
                state.velocity,
                state.radius,
                self.edge_to_edge,
                self.is_touching,
                self.is_nearby,
                self.n_all_colliding_ids,
                self.all_colliding_ids,
                self.n_direct_contacts,
                self.direct_contacts,
                self.n_nearby_contacts,
                self.nearby_contacts
        )