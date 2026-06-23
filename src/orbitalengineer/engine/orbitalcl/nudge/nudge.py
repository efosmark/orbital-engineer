import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "nudge/nudge.cl"

class NudgePipeline(CLPipelineStep):
    
    def initialize(self):
        self._apply_nudge = self._load_kernel("apply_nudge", KERNEL_FILE_LOCATION)
    
        self._position_intermediate = np.zeros(self.N, dtype=np.complex64)
        self._position_intermediate_cl = self._create_buffer(self._position_intermediate)

    def __call__(self, position: cl.Buffer, mass: cl.Buffer, distance_edge: cl.Buffer):
        self.tr.add("apply_nudge",
            self._apply_nudge(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                            
                # Args
                np.uint32(self.N),
                position,
                mass,
                distance_edge,
                self._position_intermediate_cl,
            )
        )
        cl.enqueue_copy(self.queue, position, self._position_intermediate_cl)
