import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/collision_group.cl"

class CollisionGroupPipeline(CLPipelineStep):
    
    def load_kernels(self):
        self._assign_collision_groups = self._load_kernel("assign_collision_groups", KERNEL_FILE_LOCATION)
        self._reduce_collision_groups = self._load_kernel("reduce_collision_groups", KERNEL_FILE_LOCATION)
    
    def init_memory(self, N:int):
        self.groups = np.zeros(N, dtype=np.uint32)
        self.groups_cl = self._create_buffer(self.groups)
    
    def assign_collision_groups(self, dt_step: float, mass: cl.Buffer, velocity_relative: cl.Buffer, distance_edge: cl.Buffer):
        return self.tr.add("assign_collision_groups",
            self._assign_collision_groups(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                mass,
                velocity_relative,
                distance_edge,
                self.groups_cl
            )
        )
    
    def reduce_collision_groups(self, mass: cl.Buffer):
        num_workgroups = (self.N // self.Lx) + 1
        result_indices = np.zeros(num_workgroups, dtype=np.uint32)
        result_buffer = self._create_buffer(result_indices)
        has_updates = True
        
        n_iterations = 0
        while has_updates:
            n_iterations += 1
            self.tr.add("reduce_collision_groups",
                self._reduce_collision_groups(
                    self.queue,
                    (self.N, ),   # global work size
                    (self.Lx, ),  # local work size
                    
                    # Args
                    np.uint32(self.N),
                    mass,
                    self.groups_cl,
                    result_buffer,
                )
            )

            cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
            has_updates = result_indices.any()
            if n_iterations > 5:
                print("TOO MANY ITERATIONS")
                raise SystemExit

    def __call__(self, dt_step:float, velocity_relative: cl.Buffer, mass: cl.Buffer, distance_edge: cl.Buffer):
        self.assign_collision_groups(dt_step, mass, velocity_relative, distance_edge)
        self.reduce_collision_groups(mass)
        