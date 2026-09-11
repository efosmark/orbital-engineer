import numpy as np
import pyopencl as cl
from orbitalengineer.engine import config
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "interaction/interaction.cl"

class InteractionPipeline(PipelineComponent):
    debug_flag = 'interaction'
    
    def initialize(self):
        self._knl_interaction_time = self._load_kernel("interaction_time", KERNEL_FILE_LOCATION)

        self.dt_until_collision = self.alloc(self.N * self.N * 2, dtype=np.float32)
        self.min_dt_per_body = self.alloc(self.N, np.float32, fill=[np.inf for i in range(self.N)])

    def minimum_viable_dt(self, dt_step):
        min_dt_per_body = self.get_host_vector(self.min_dt_per_body, sync=True)
        try:
            min_toi = min([t for t in min_dt_per_body if t > 0])
        except ValueError:
            min_toi = dt_step
        return max(min(dt_step, min_toi), config.EPS_TIME)

    def compute_interaction_time(self, dt_step:float, state:PrimaryStateVectors):
        min_dt_per_body = self.get_host_vector(self.min_dt_per_body)
        min_dt_per_body[:] = 0.0
        self.sync_to_device(self.min_dt_per_body)

        
        return self._knl_interaction_time(
                self.queue,
                state.grid_stride_global_size,
                state.grid_stride_local_size,
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                state.flags,
                state.position,
                state.velocity,
                state.radius,
                self.dt_until_collision,
                self.min_dt_per_body,
        )
    
    def __call__(self, dt_step:float, state:PrimaryStateVectors):
        self.compute_interaction_time(dt_step, state)