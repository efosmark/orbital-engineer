import numpy as np
import pyopencl as cl
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "nudge/nudge.cl"

class NudgePipeline(PipelineComponent):
    
    def initialize(self):
        self._apply_nudge = self._load_kernel("apply_nudge", KERNEL_FILE_LOCATION)
        self._position_intermediate = self.vec.alloc(self.vec.N, dtype=np.complex64)

    def __call__(self, state:PrimaryStateVectors):
        self._apply_nudge(
            self.queue,
            self.vec.grid_stride_global_size,
            self.vec.grid_stride_local_size,
            
            # Args
            np.uint32(self.vec.N_bodies_alloc),
            np.uint32(self.vec.N_bodies_valid),
            state.flags,
            state.position,
            state.mass,
            state.radius,
            self._position_intermediate,
        )
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.position, self._position_intermediate))
