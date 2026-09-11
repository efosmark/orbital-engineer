import pyopencl as cl
import numpy as np

from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE = "velocity/velocity.cl"

class VelocityPipeline(PipelineComponent):
    debug_flag = "velocity"
    
    def initialize(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
        
        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)
        self.force = self.alloc(self.N * self.N, dtype=np.complex64, shared_name="force")
    
    def compute_velocity(self, dt_step:float, state:PrimaryStateVectors, contacting: FindContactingBodiesPipeline):
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
            self._velocity_intermediate,
            self.force
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)

    def __call__(self, dt_step:float, state:PrimaryStateVectors, contacting:FindContactingBodiesPipeline):
        self.compute_velocity(dt_step, state, contacting)