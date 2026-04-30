import numpy as np
from orbitalengineer.engine.cl.dimension import CLPipelineStep


class AccelerationStep(CLPipelineStep):
    
    def load_kernels(self):
        self._kernel = self._load_kernel("compute_acceleration", "kernel/acceleration.cl")
    
    def init_memory(self, N:int):
        self._accel_host = np.zeros(N * N, dtype=np.complex64)
        self.cl = self._create_buffer(self._accel_host)

    def __call__(self, dt_step, position, mass, radius):
        Lx = 256
        return self._kernel(
            self.queue,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            np.float32(dt_step),
            position,
            mass,
            radius,
            self.cl
        )