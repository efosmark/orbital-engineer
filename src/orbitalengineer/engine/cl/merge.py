import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/merge.cl"

class MergePipeline(CLPipelineStep):
    debug_flag = "merge"

    def initialize(self):
        self._assign_merge_groups = self._load_kernel("collide_merge_group_assign", KERNEL_FILE_LOCATION)
        self._collide_merge_group_reduce = self._load_kernel("collide_merge_group_reduce", KERNEL_FILE_LOCATION)
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
        
        self._groups = np.zeros(self.N, dtype=np.uint32)
        self._groups_cl = self._create_buffer(self._groups)
    
    def collide_merge_group_assign(self, dt_step: float, status: cl.Buffer, mass: cl.Buffer, velocity_relative: cl.Buffer, distance_edge: cl.Buffer):
        return self.tr.add("collide_merge_group_assign",
            self._assign_merge_groups(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                status,
                mass,
                velocity_relative,
                distance_edge,
                self._groups_cl
            )
        )
    
    def collide_merge_group_reduce(self, status: cl.Buffer, mass: cl.Buffer):
        num_workgroups = (self.N // self.Lx) + 1
        result_indices = np.zeros(num_workgroups, dtype=np.uint32)
        result_buffer = self._create_buffer(result_indices)
        has_updates = True
        
        n_iterations = 0
        while has_updates:
            n_iterations += 1
            self.tr.add("collide_merge_group_reduce", self._collide_merge_group_reduce(
                self.queue,
                (self.N, ),   # global work size
                (self.Lx, ),  # local work size
                
                # Args
                np.uint32(self.N),
                status,
                mass,
                self._groups_cl,
                result_buffer,
            ))

            cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
            has_updates = result_indices.any()
            if n_iterations > 5:
                print("TOO MANY ITERATIONS")
                raise SystemExit
        
            
    def compute_merging_collision(self, status: cl.Buffer, merge_group: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        return self.tr.add("collide_merge",
            self._compute_merging_collision(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                status,
                merge_group,
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

    def __call__(self, dt_step, flags: cl.Buffer, velocity_relative:cl.Buffer, edge_distance: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, mass: cl.Buffer, radius: cl.Buffer):
        self.collide_merge_group_assign(dt_step, flags, mass, velocity_relative, edge_distance)
        self.collide_merge_group_reduce(flags, mass)
        self.compute_merging_collision(flags, self._groups_cl, position, velocity, mass, radius)
        
        cl.enqueue_copy(self.queue, flags,    self._flags_intermediate_cl)
        cl.enqueue_copy(self.queue, position, self._position_intermediate_cl)
        cl.enqueue_copy(self.queue, velocity, self._velocity_intermediate_cl)
        cl.enqueue_copy(self.queue, mass,     self._mass_intermediate_cl)
        cl.enqueue_copy(self.queue, radius,   self._radius_intermediate_cl)
