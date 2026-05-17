import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/hill_radius.cl"

class HillRadiusPipeline(CLPipelineStep):
    
    def initialize(self):
        self._compute_hill_radius = self._load_kernel("compute_hill_radius", KERNEL_FILE_LOCATION)

        self._hill_radius = np.zeros(self.N * self.N, dtype=np.float32)
        self._hill_radius_cl = self._create_buffer(self._hill_radius)

        self._minimum_hill_radius = np.zeros(self.N, dtype=np.float32)
        self._minimum_hill_radius_cl = self._create_buffer(self._minimum_hill_radius)


    def __call__(self, position: cl.Buffer, mass: cl.Buffer, radius:cl.Buffer):
        self.tr.add("hill_radius",
            self._compute_hill_radius(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                            
                # Args
                np.uint32(self.N),
                position,
                mass,
                radius,
                self._hill_radius_cl,
                self._minimum_hill_radius_cl
            )
        )