import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/bounce.cl"

class BouncePipeline(CLPipelineStep):
    
    def load_kernels(self):
        self._compute_bouncing_collision = self._load_kernel("compute_bouncing_collision", KERNEL_FILE_LOCATION)
    
    def __call__(self, velocity_relative: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, distance_edge: cl.Buffer):
        return self.tr.add("compute_bouncing_collision",
            self._compute_bouncing_collision(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                            
                # Args
                np.uint32(self.N),
                velocity_relative,
                position,
                velocity,
                mass,
                distance_edge
            ))
        