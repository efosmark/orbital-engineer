import pyopencl as cl
import numpy as np

from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.distance.distance import DistancePipeline
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

KERNEL_FILE = "velocity/velocity.cl"

class VelocityPipeline(PipelineComponent):
    debug_flag = "velocity"
    Lx = 256
    
    def initialize(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
        self._velocity_intermediate = self.vec.alloc(self.vec.N, dtype=np.complex64)
        self.force = self.vec.alloc(self.vec.N * self.vec.N, dtype=np.complex64, shared_name="force")

    
    def compute_velocity(self, dt_step:float, state:PrimaryStateVectors, distance: DistancePipeline):
        self._compute_velocity(
            self.queue,
            self.vec.grid_stride_global_size,
            self.vec.grid_stride_local_size,
            
            # Args
            np.uint32(self.vec.N_bodies_alloc),
            np.uint32(self.vec.N_bodies_valid),
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
