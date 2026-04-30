import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/merge.cl"

class Merge(CLPipelineStep):
    Lx = 256
    
    def load_kernels(self):
        self._assign_collision_groups = self._load_kernel("assign_collision_groups", KERNEL_FILE_LOCATION)
        self._reduce_collision_groups = self._load_kernel("reduce_collision_groups", KERNEL_FILE_LOCATION)
        self._compute_merging_collision = self._load_kernel("compute_merging_collision", KERNEL_FILE_LOCATION)
        self._apply_merge = self._load_kernel("apply_merge", KERNEL_FILE_LOCATION)
    
    def init_memory(self, N:int):
        self._group = np.zeros(N, dtype=np.uint32)
        self._group_cl = self._create_buffer(self._group)
        
        self._mass_partial = np.zeros(N, dtype=np.float32)
        self._mass_partial_cl = self._create_buffer(self._mass_partial)
        
        self._velocity_partial = np.zeros(N * N, dtype=np.complex64)
        self._velocity_partial_cl = self._create_buffer(self._velocity_partial)
        
        self._position_partial = np.zeros(N * N, dtype=np.complex64)
        self._position_partial_cl = self._create_buffer(self._position_partial)

    def assign_collision_groups(self, mass, velocity_relative, distance_edge):
        return self._assign_collision_groups(
            self.queue,
            (self.N * self.Lx, ),  # global work size
            (self.Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            mass,
            velocity_relative,
            distance_edge,
            self._group_cl
        )
    
    def reduce_collision_groups(self, mass):
        num_workgroups = (self.N // self.Lx) + 1
        result_indices = np.zeros(num_workgroups, dtype=np.uint32)
        result_buffer = self._create_buffer(result_indices)
        has_updates = True
        
        n_iterations = 0
        while has_updates:
            n_iterations += 1
            self._reduce_collision_groups(
                self.queue,
                (self.N, ),   # global work size
                (self.Lx, ),  # local work size
                
                # Args
                np.uint32(self.N),
                mass,
                self._group_cl,
                result_buffer,
            )

            cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
            has_updates = result_indices.any()
            if n_iterations > 5:
                print("TOO MANY ITERATIONS")
                raise SystemExit
        #raise SystemExit
        
    def compute_merging_collision(self, position, velocity, mass):
        return self._compute_merging_collision(
            self.queue,
            (self.N * self.Lx, ),  # global work size
            (self.Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            self._group_cl,
            position,
            velocity,
            mass,
            self._position_partial_cl,
            self._velocity_partial_cl,
            self._mass_partial_cl
        )
    
    def apply_merge(self, position, velocity, mass, radius):
        return self._apply_merge(
            self.queue,
            (self.N, ),  # global work size
            None,
            
            # Args
            self._mass_partial_cl,
            self._velocity_partial_cl,
            self._position_partial_cl,
            mass,
            velocity,
            position,
            radius
        )

    def __call__(self, velocity_relative: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer, distance_edge: cl.Buffer):
        self.assign_collision_groups(mass, velocity_relative, distance_edge)
        self.reduce_collision_groups(mass)
        self.compute_merging_collision(position, velocity, mass)
        self.apply_merge(position, velocity, mass, radius)
        