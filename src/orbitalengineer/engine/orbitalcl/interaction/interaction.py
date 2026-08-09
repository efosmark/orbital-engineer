import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep

KERNEL_FILE_LOCATION = "interaction/interaction.cl"

class InteractionGroupPipeline(CLPipelineStep):
    debug_flag = 'interaction'
    
    def initialize(self):
        self._compute_interaction_time = self._load_kernel("compute_interaction_time", KERNEL_FILE_LOCATION)
        #self._assign_interaction_groups = self._load_kernel("assign_interaction_groups", KERNEL_FILE_LOCATION)
        #self._reduce_interaction_groups = self._load_kernel("reduce_interaction_groups", KERNEL_FILE_LOCATION)
        #self._collect_group_members = self._load_kernel("collect_group_members", KERNEL_FILE_LOCATION)
    
        self.toi = np.zeros(self.N * self.N * 2, dtype=np.float32)
        self.toi_cl = self._create_buffer(self.toi)

        self.node_dt = np.array([np.inf for i in range(self.N)], dtype=np.float32)
        self.node_dt_cl = self._create_buffer(self.node_dt)
        
        self.group_dt = np.array([np.inf for i in range(self.N)], dtype=np.float32)
        self.group_dt_cl = self._create_buffer(self.group_dt)
        
        self.groups = np.zeros(self.N, dtype=np.uint32)
        self.groups_cl = self._create_buffer(self.groups)    
    
    def compute_interaction_time(self, flags: cl.Buffer, position: cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer):
        return self.tr.add("interaction_time",
            self._compute_interaction_time(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                flags,
                position,
                velocity,
                radius,
                self.toi_cl,
                self.node_dt_cl
            )
        )
    
    # def assign_interaction_groups(self, dt_step: float, mass: cl.Buffer):
    #     return self.tr.add("assign_interaction_groups",
    #         self._assign_interaction_groups(
    #             self.queue,
    #             (self.N * self.Lx, ),  # global work size
    #             (self.Lx, ),           # local work size
                
    #             # Args
    #             np.uint32(self.N),
    #             np.float32(dt_step),
    #             mass,
    #             self.toi_cl,
    #             self.groups_cl
    #         )
    #     )
    
    # def reduce_interaction_groups(self, mass: cl.Buffer):
    #     num_workgroups = (self.N // self.Lx) + 1
    #     result_indices = np.zeros(num_workgroups, dtype=np.uint32)
    #     result_buffer = self._create_buffer(result_indices)
    #     has_updates = True
        
    #     n_iterations = 0
    #     while has_updates:
    #         n_iterations += 1
    #         self.tr.add("reduce_interaction_groups",
    #             self._reduce_interaction_groups(
    #                 self.queue,
    #                 (self.N, ),   # global work size
    #                 (self.Lx, ),  # local work size
                    
    #                 # Args
    #                 np.uint32(self.N),
    #                 mass,
    #                 self.node_dt_cl,
    #                 self.group_dt_cl,
    #                 self.groups_cl,
    #                 result_buffer,
    #             ))

    #         cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
    #         has_updates = result_indices.any()
    #         if n_iterations > 5: raise SystemExit

    # def collect_group_members(self ):
    #     return self.tr.add("collect_group_members",
    #         self._collect_group_members(
    #             self.queue,
    #             (self.N * self.Lx, ),  # global work size
    #             None, #(self.Lx, ),           # local work size
                
    #             # Args
    #             np.uint32(self.N),
    #             self.groups_cl,
    #             self.group_members_cl
    #         ))

    def __call__(self, dt_step:float, flags:cl.Buffer, position:cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, mass: cl.Buffer):
        self.compute_interaction_time(flags, position, velocity, radius)
        #self.assign_interaction_groups(dt_step, mass)
        #self.reduce_interaction_groups(mass)

        #cl.enqueue_copy(self.queue, self.node_dt, self.node_dt_cl).wait()
        #print("self.node_dt", self.node_dt)
        
        #cl.enqueue_copy(self.queue, self.group_dt, self.group_dt_cl).wait()
        #print("self.group_dt", self.group_dt)
