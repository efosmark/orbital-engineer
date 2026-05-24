import pyopencl as cl
import numpy as np
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE = "position/position.cl"

class PositionPipeline(CLPipelineStep):
    debug_flag = "position"
    
    def initialize(self):
        self._compute_position = self._load_kernel('compute_position', KERNEL_FILE)
    
    def compute_position(self, dt_step, status: cl.Buffer, velocity:cl.Buffer, position:cl.Buffer):
        self.tr.add("position",
            self._compute_position(
                self.queue,
                (self.N,), # global work size
                None,      # local work size
                            
                # Args
                np.float32(dt_step),
                status,
                velocity,
                position
            ))

    def __call__(self, dt_step, status: cl.Buffer, velocity: cl.Buffer, position: cl.Buffer):
        self.compute_position(dt_step, status, velocity, position)
        