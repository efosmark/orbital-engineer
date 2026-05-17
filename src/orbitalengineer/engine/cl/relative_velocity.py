import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE = "kernel/relative_velocity.cl"

class RelativeVelocityPipeline(CLPipelineStep):
    debug_flag = "relative_velocity"
    
    def initialize(self):
        self._compute_relative_velocity = self._load_kernel('compute_relative_velocity', KERNEL_FILE)
        
        self.grid_stride_global_work_size = (self.N * self.Lx, )
        self.grid_stride_local_work_size = (self.Lx, )
    
    def compute_relative_velocity(self, position: cl.Buffer, velocity: cl.Buffer, relative_velocity: cl.Buffer):
        return self.tr.add("relative_velocity",
            self._compute_relative_velocity(
                self.queue,
                self.grid_stride_global_work_size,
                self.grid_stride_local_work_size,
                
                # Args
                np.uint32(self.N),
                position,
                velocity,
                relative_velocity,
            ))

    def __call__(self, position: cl.Buffer, velocity: cl.Buffer , relative_velocity: cl.Buffer):
        self.compute_relative_velocity(position, velocity, relative_velocity)