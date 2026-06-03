import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE = "velocity/velocity.cl"

class VelocityPipeline(CLPipelineStep):
    debug_flag = "velocity"
    
    def initialize(self):
        self._compute_velocity = self._load_kernel('compute_velocity', KERNEL_FILE)
        
        self._acceleration = np.zeros(self.N * self.N, dtype=np.complex64)
        self._acceleration_cl = self._create_buffer(self._acceleration)
        
        self.grid_stride_global_work_size = (self.N * self.Lx, )
        self.grid_stride_local_work_size = (self.Lx, )
    
    def compute_velocity(self, dt_step:float, status: cl.Buffer, position: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer, distance_edge: cl.Buffer, velocity: cl.Buffer):
        return self.tr.add("velocity",
            self._compute_velocity(
                self.queue,
                self.grid_stride_global_work_size,
                self.grid_stride_local_work_size,
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                status,
                position,
                mass,
                radius,
                distance_edge,
                self._acceleration_cl,
                velocity
            ))

    def __call__(self, dt_step:float, status: cl.Buffer, position: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer, distance_edge: cl.Buffer, velocity: cl.Buffer):
        self.compute_velocity(dt_step, status, position, mass, radius, distance_edge, velocity)
        cl.enqueue_copy(self.queue, self._acceleration, self._acceleration_cl)