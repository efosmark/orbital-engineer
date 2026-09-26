import pyopencl as cl
import numpy as np

from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE = "velocity/velocity.cl"

class VelocityPipeline(PipelineComponent):
    debug_flag = "velocity"
    
    def initialize(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
        self._compute_velocity_pairwise = self._load_kernel('compute_velocity_pairwise', KERNEL_FILE)
        self._apply_velocity_updates = self._load_kernel('apply_velocity_updates', KERNEL_FILE)
        
        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)
        self.force = self.alloc(self.N * self.N, dtype=np.complex64, shared_name="force")
    
        self._velocity_updates = self.alloc(self.N * self.N, dtype=np.complex64)
        self._create_pairs()
        
    
    def _create_pairs(self):
        pair_dtype = np.dtype([
            ("idx",    np.uint32),  # Pairwise index
            ("i",      np.uint32),  #  
            ("j",      np.uint32),
        ], align=True)

        self.ix, self.jx = np.triu_indices(int(self.N), k=1)
        self.num_pairs = self.ix.size
        self.pairs_host = np.zeros(self.ix.size, dtype=pair_dtype)
        for x in range(self.pairs_host.size):
            self.pairs_host["idx"][x] = x
            self.pairs_host["i"][x] = self.ix[x]
            self.pairs_host["j"][x] = self.jx[x]
        self.pairs = self._create_buffer(self.pairs_host)
    
    def compute_velocity(self, dt_step:float, state:PrimaryStateVectors, distance: DistancePipeline):
        self._compute_velocity(
            self.queue,
            state.grid_stride_global_size,
            state.grid_stride_local_size,
            
            # Args
            np.uint32(state.N),
            np.float32(dt_step),
            state.flags,
            state.position,
            state.mass,
            state.radius,
            state.velocity,
            distance.is_touching,
            self._velocity_intermediate,
            self.force
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)

    def compute_velocity_pairwise(self, dt_step:float, state:PrimaryStateVectors):
        self._compute_velocity_pairwise(
            self.queue,
            (self.ix.size, ),
            None,
            
            # Args
            np.uint32(state.N),
            np.float32(dt_step),
            self.pairs,
            state.flags,
            state.position,
            state.mass,
            state.radius,
            state.velocity,
            self._velocity_updates
        )
        
        self._apply_velocity_updates(
            self.queue,
            state.grid_stride_global_size,
            state.grid_stride_local_size,
            
            # Args
            np.uint32(state.N),
            state.velocity,
            self._velocity_updates,
            self._velocity_intermediate
        )
        
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)


    def __call__(self, dt_step:float, state:PrimaryStateVectors, distance: DistancePipeline):
        self.compute_velocity(dt_step, state, distance)
        #self.compute_velocity_pairwise(dt_step, state)