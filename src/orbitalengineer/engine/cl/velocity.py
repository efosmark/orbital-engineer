import pyopencl as cl
import numpy as np
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE = "kernel/velocity.cl"

class VelocityPipeline(CLPipelineStep):
    
    def load_kernels(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
    
    def init_memory(self, N:int, Lx:int|None=None):
        self.N = N
        if Lx is not None:
            self.Lx = Lx
        
        self.grid_stride_global_work_size = (self.N * self.Lx, )
        self.grid_stride_local_work_size = (self.Lx, )
    
    def compute_velocity(self, dt_step:float, position: cl.Buffer, mass: cl.Buffer, velocity: cl.Buffer):
        return self.tr.add("compute_velocity",
            self._compute_velocity(
                self.queue,
                self.grid_stride_global_work_size,
                self.grid_stride_local_work_size,
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                position,
                mass,
                velocity
            ))

    def __call__(self, dt_step:float, position: cl.Buffer, mass: cl.Buffer, velocity: cl.Buffer):
        self.compute_velocity(dt_step, position, mass, velocity)