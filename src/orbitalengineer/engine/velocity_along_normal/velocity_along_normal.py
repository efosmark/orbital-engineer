import numpy as np
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "velocity_along_normal/velocity_along_normal.cl"

class VelocityAlongNormalPipeline(PipelineComponent):
    debug_flag = 'velocity_along_normal'
    
    def initialize(self):
        self._knl_velocity_along_normal = self._load_kernel("velocity_along_normal", KERNEL_FILE_LOCATION)
        self.velocity_along_normal = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.float32)
    
    def compute_velocity_along_normal(self, state:PrimaryStateVectors):
        return self._knl_velocity_along_normal(
                self.queue,
                (self.vec.N_bodies_valid * self.vec.Lx, ),
                (self.vec.Lx, ),
                
                # Args
                np.uint32(self.vec.N_bodies_alloc),
                np.uint32(self.vec.N_bodies_valid),
                state.flags,
                state.position,
                state.velocity,
                self.velocity_along_normal,
        )
    
    def __call__(self, state:PrimaryStateVectors):
        self.compute_velocity_along_normal(state)