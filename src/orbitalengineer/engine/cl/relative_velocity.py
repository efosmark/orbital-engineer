import pyopencl as cl
import numpy as np
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE = "kernel/relative_velocity.cl"

class RelativeVelocityPipeline(CLPipelineStep):
    
    def load_kernels(self):
        self._compute_relative_velocity = self._load_kernel('compute_relative_velocity', KERNEL_FILE)
    
    def init_memory(self, N:int, Lx:int|None=None):
        self.N = N
        if Lx is not None:
            self.Lx = Lx
        
        self.grid_stride_global_work_size = (self.N * self.Lx, )
        self.grid_stride_local_work_size = (self.Lx, )
    
    def compute_relative_velocity(self, position: cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, relative_velocity: cl.Buffer):
        return self.tr.add("compute_relative_velocity",
            self._compute_relative_velocity(
                self.queue,
                self.grid_stride_global_work_size,
                self.grid_stride_local_work_size,
                
                # Args
                np.uint32(self.N),
                position,
                velocity,
                radius,
                relative_velocity,
            ))

    def __call__(self, position: cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, relative_velocity: cl.Buffer):
        self.compute_relative_velocity(position, velocity, radius, relative_velocity)