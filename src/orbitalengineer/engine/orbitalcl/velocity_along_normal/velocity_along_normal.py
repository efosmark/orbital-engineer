import numpy as np
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "velocity_along_normal/velocity_along_normal.cl"

class VelocityAlongNormalPipeline(PipelineComponent):
    debug_flag = 'velocity_along_normal'
    
    def initialize(self):
        self._knl_velocity_along_normal = self._load_kernel("velocity_along_normal", KERNEL_FILE_LOCATION)
        self.velocity_along_normal = self.alloc(self.N * self.N, dtype=np.float32)
    
    def compute_velocity_along_normal(self, state:PrimaryStateVectors):
        return self._knl_velocity_along_normal(
                self.queue,
                state.grid_stride_global_size,
                state.grid_stride_local_size,
                
                # Args
                np.uint32(self.N),
                state.flags,
                state.position,
                state.velocity,
                self.velocity_along_normal,
        )
    
    def __call__(self, state:PrimaryStateVectors):
        self.compute_velocity_along_normal(state)