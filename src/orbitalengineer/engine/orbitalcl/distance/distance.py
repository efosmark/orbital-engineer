import numpy as np
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "distance/distance.cl"

class DistancePipeline(PipelineComponent):
    debug_flag = 'distance'
    
    def initialize(self):
        self._knl_edge_distance = self._load_kernel("edge_distance", KERNEL_FILE_LOCATION)
        self.edge_to_edge = self.alloc(self.N * self.N, dtype=np.float32)
        self.is_touching = self.alloc(self.N * self.N, dtype=np.bool)
    
    def __call__(self, state:PrimaryStateVectors):
        self._knl_edge_distance(
                self.queue,
                state.grid_stride_global_size,
                state.grid_stride_local_size,
                
                # Args
                np.uint32(self.N),
                state.flags,
                state.position,
                state.velocity,
                state.radius,
                self.edge_to_edge,
                self.is_touching
        )