import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/merge.cl"

class MergePipeline(CLPipelineStep):
    debug_flag = "merge"

    def initialize(self):
        self._compute_merging_collision = self._load_kernel("compute_merging_collision", KERNEL_FILE_LOCATION)
        self._apply_merge = self._load_kernel("apply_merge", KERNEL_FILE_LOCATION)
    
        self._mass_intermediate = np.zeros(self.N, dtype=np.float32)
        self._mass_intermediate_cl = self._create_buffer(self._mass_intermediate)
        
        self._velocity_intermediate = np.zeros(self.N * self.N, dtype=np.complex64)
        self._velocity_intermediate_cl = self._create_buffer(self._velocity_intermediate)
        
        self._position_intermediate = np.zeros(self.N * self.N, dtype=np.complex64)
        self._position_intermediate_cl = self._create_buffer(self._position_intermediate)
        
    def compute_merging_collision(self, status: cl.Buffer, collision_group: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer):
        return self.tr.add("collide_merge",
            self._compute_merging_collision(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                status,
                collision_group,
                position,
                velocity,
                mass,
                self._position_intermediate_cl,
                self._velocity_intermediate_cl,
                self._mass_intermediate_cl
            ))
    
    def apply_merge(self, status: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        return self.tr.add("collide_merge_apply",
            self._apply_merge(
                self.queue,
                (self.N, ),  # global work size
                None,
                
                # Args
                status,
                self._mass_intermediate_cl,
                self._velocity_intermediate_cl,
                self._position_intermediate_cl,
                mass,
                velocity,
                position,
                radius
            )
        )

    def __call__(self, status: cl.Buffer, collision_group: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        self.compute_merging_collision(status, collision_group, position, velocity, mass)
        self.apply_merge(status, position, velocity, mass, radius)
        