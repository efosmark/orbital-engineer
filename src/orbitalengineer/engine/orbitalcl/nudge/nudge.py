import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "nudge/nudge.cl"

class NudgePipeline(PipelineComponent):
    
    def initialize(self):
        self._apply_nudge = self._load_kernel("apply_nudge", KERNEL_FILE_LOCATION)
        self._position_intermediate = self.alloc(self.N, dtype=np.complex64)

    def __call__(self, state:PrimaryStateVectors):
        self._apply_nudge(
            self.queue,
            (self.N * self.Lx, ),  # global work size
            (self.Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            state.flags,
            state.position,
            state.mass,
            state.radius,
            self._position_intermediate,
        )
        cl.enqueue_copy(self.queue, state.position, self._position_intermediate)
