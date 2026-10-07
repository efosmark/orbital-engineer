import numpy as np
from orbitalengineer.engine import config
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "interaction/interaction.cl"

class InteractionPipeline(PipelineComponent):
    debug_flag = 'interaction'
    
    def initialize(self):
        self._knl_compute_interaction = self._load_kernel("compute_interaction", KERNEL_FILE_LOCATION)

        self.dt_until_collision = self.vec.alloc(self.vec.N * self.vec.N * 2, dtype=np.float32)
        self.min_dt_per_body = self.vec.alloc(self.vec.N, np.float32, fill=[np.inf for i in range(self.vec.N)])
    
    def minimum_viable_dt(self, dt_step, eps_time):
        with self.tr('minimum_viable_dt (host)'):
            min_dt_per_body = self.vec.get_host_vector(self.min_dt_per_body, sync=True)
            try:
                min_toi = np.min(min_dt_per_body[min_dt_per_body > 0])
            except ValueError:
                min_toi = dt_step
            result = max(min(dt_step, min_toi), eps_time)
        return result      

    def compute_interaction(self, dt_step:float, state:PrimaryStateVectors):
        min_dt_per_body = self.vec.get_host_vector(self.min_dt_per_body)
        min_dt_per_body[:] = 0.0
        self.vec.sync_to_device(self.min_dt_per_body)
        
        return self._knl_compute_interaction(
                self.queue,
                (self.vec.N_bodies_valid * self.vec.Lx, ),
                (self.vec.Lx, ),
                
                # Args
                np.uint32(self.vec.N_bodies_alloc),
                np.uint32(self.vec.N_bodies_valid),
                np.float32(dt_step),
                state.flags,
                state.position,
                state.velocity,
                state.radius,
                self.dt_until_collision,
                self.min_dt_per_body,
        )
    
    def __call__(self, dt_step:float, state:PrimaryStateVectors):
        self.compute_interaction(dt_step, state)