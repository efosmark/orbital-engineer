import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "merge/merge.cl"

class MergePipeline(CLPipelineStep):
    debug_flag = "merge"

    def initialize(self):
        self._compute_merging_collision = self._load_kernel("compute_merging_collision", KERNEL_FILE_LOCATION)
    
        self._mass_intermediate = np.zeros(self.N, dtype=np.float32)
        self._mass_intermediate_cl = self._create_buffer(self._mass_intermediate)
        
        self._radius_intermediate = np.zeros(self.N, dtype=np.float32)
        self._radius_intermediate_cl = self._create_buffer(self._radius_intermediate)
        
        self._velocity_intermediate = np.zeros(self.N, dtype=np.complex64)
        self._velocity_intermediate_cl = self._create_buffer(self._velocity_intermediate)
        
        self._position_intermediate = np.zeros(self.N, dtype=np.complex64)
        self._position_intermediate_cl = self._create_buffer(self._position_intermediate)
        
        self._flags_intermediate = np.zeros(self.N, dtype=np.uint32)
        self._flags_intermediate_cl = self._create_buffer(self._flags_intermediate)
        
        self._groups = np.arange(self.N, dtype=np.uint32)
        self._groups_cl = self._create_buffer(self._groups)
        self._groups_prev = np.arange(self.N, dtype=np.uint32)


    def compute_merging_collision(self, status: cl.Buffer, cgroup: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        return self.tr.add("collide_merge",
            self._compute_merging_collision(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                status,
                cgroup,
                position,
                velocity,
                mass,
                radius,
                self._flags_intermediate_cl,
                self._position_intermediate_cl,
                self._velocity_intermediate_cl,
                self._mass_intermediate_cl,
                self._radius_intermediate_cl
            ))
    
    def __call__(self, flags: cl.Buffer, cgroup: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        self.compute_merging_collision(flags, cgroup, position, velocity, mass, radius)
        cl.enqueue_copy(self.queue, flags,    self._flags_intermediate_cl)
        cl.enqueue_copy(self.queue, position, self._position_intermediate_cl)
        cl.enqueue_copy(self.queue, velocity, self._velocity_intermediate_cl)
        cl.enqueue_copy(self.queue, mass,     self._mass_intermediate_cl)
        cl.enqueue_copy(self.queue, radius,   self._radius_intermediate_cl)

    def find_merged_bodies(self):
        """Finds the merge events that happened since the last time called."""
        cl.enqueue_copy(self.queue,  self._groups, self._groups_cl)
        diff = np.argwhere(self._groups != self._groups_prev)
        if len(diff) > 0: diff = diff[0]        
        merged_ids = np.column_stack((diff, self._groups[diff]))
        self._groups_prev[:] = self._groups
        return merged_ids