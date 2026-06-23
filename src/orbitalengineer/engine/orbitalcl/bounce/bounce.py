import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "bounce/bounce.cl"

class BouncePipeline(CLPipelineStep):
    debug_flag = "bounce"
    
    def initialize(self):
        self._compute_bouncing_collision = self._load_kernel("compute_bouncing_collision", KERNEL_FILE_LOCATION)

        self.collision_point = np.zeros(self.N*self.N, dtype=np.complex64)
        self.collision_point_cl = self._create_buffer(self.collision_point)

        self._velocity_intermediate = np.zeros(self.N, dtype=np.complex64)
        self._velocity_intermediate_cl = self._create_buffer(self._velocity_intermediate)

        self._position_intermediate = np.zeros(self.N, dtype=np.complex64)
        self._position_intermediate_cl = self._create_buffer(self._position_intermediate)


    def __call__(self, status: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer, velocity_relative: cl.Buffer, distance_edge: cl.Buffer):
        self.tr.add("collide_bounce",
            self._compute_bouncing_collision(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                            
                # Args
                np.uint32(self.N),
                status,
                position,
                velocity,
                mass,
                radius,
                velocity_relative,
                distance_edge,
                self._velocity_intermediate_cl,
                self._position_intermediate_cl,
                self.collision_point_cl
            ))
        
        cl.enqueue_copy(self.queue, velocity, self._velocity_intermediate_cl)
        #cl.enqueue_copy(self.queue, position, self._position_intermediate_cl)
