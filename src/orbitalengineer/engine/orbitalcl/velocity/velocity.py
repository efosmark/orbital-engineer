import pyopencl as cl
import numpy as np

from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE = "velocity/velocity.cl"

class VelocityPipeline(PipelineComponent):
    debug_flag = "velocity"
    Lx = 256
    
    def initialize(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)
        self.force = self.alloc(self.N * self.N, dtype=np.complex64, shared_name="force")

        self.grid_stride_global_size = (self.N * self.Lx, )
        self.grid_stride_local_size = (self.Lx, )
    
    def compute_velocity(self, dt_step:float, state:PrimaryStateVectors, distance: DistancePipeline):
        self._compute_velocity(
            self.queue,
            self.grid_stride_global_size,
            self.grid_stride_local_size,
            
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
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate))


    def __call__(self, dt_step:float, state:PrimaryStateVectors, distance: DistancePipeline):
        self.compute_velocity(dt_step, state, distance)
