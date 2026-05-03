import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "kernel/interaction.cl"

class InteractionGroupPipeline(CLPipelineStep):
    
    def load_kernels(self):
        self._compute_interaction_time = self._load_kernel("compute_interaction_time", KERNEL_FILE_LOCATION)
        self._assign_interaction_groups = self._load_kernel("assign_interaction_groups", KERNEL_FILE_LOCATION)
        self._reduce_interaction_groups = self._load_kernel("reduce_interaction_groups", KERNEL_FILE_LOCATION)
    
    def init_memory(self, N:int):
        self.groups = np.zeros(N, dtype=np.uint32)
        self.groups_cl = self._create_buffer(self.groups)
    
    def compute_interaction_time(self, position: cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, toi: cl.Buffer, min_impact_time: cl.Buffer):
        return self.tr.add("compute_interaction_time",
            self._compute_interaction_time(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                position,
                velocity,
                radius,
                toi,
                min_impact_time
            )
        )
    
    def assign_interaction_groups(self, dt_step: float, mass: cl.Buffer, impact_times: cl.Buffer):
        return self.tr.add("assign_interaction_groups",
            self._assign_interaction_groups(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                mass,
                impact_times,
                self.groups_cl
            )
        )
    
    def reduce_interaction_groups(self, mass: cl.Buffer):
        num_workgroups = (self.N // self.Lx) + 1
        result_indices = np.zeros(num_workgroups, dtype=np.uint32)
        result_buffer = self._create_buffer(result_indices)
        has_updates = True
        
        n_iterations = 0
        while has_updates:
            n_iterations += 1
            self.tr.add("reduce_interaction_groups",
                self._reduce_interaction_groups(
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

    def __call__(self, dt_step:float, position:cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, toi: cl.Buffer, min_impact_time:cl.Buffer, mass: cl.Buffer):
        self.compute_interaction_time(position, velocity, radius, toi, min_impact_time)
        self.assign_interaction_groups(dt_step, mass, toi)
        self.reduce_interaction_groups(mass)
        