import pyopencl as cl
import numpy as np
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE = "position/position.cl"

class PositionPipeline(PipelineComponent):
    debug_flag = "position"
    
    def initialize(self):
        self._compute_position = self._load_kernel('compute_position', KERNEL_FILE)
    
    def compute_position(self, dt_step, state:PrimaryStateVectors):
        self._compute_position(
            self.queue,
            (self.N,), # global work size
            None,      # local work size
            
            # Args
            np.float32(dt_step),
            state.flags,
            state.velocity,
            state.position
        )

    def __call__(self, dt_step, state:PrimaryStateVectors):
        self.compute_position(dt_step, state)
        