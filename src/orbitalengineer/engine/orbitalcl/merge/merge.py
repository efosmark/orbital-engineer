import numpy as np
import pyopencl as cl
from orbitalengineer.engine import config
from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "merge/merge.cl"

class MergePipeline(PipelineComponent):
    debug_flag = "merge"

    def initialize(self):
        self._compute_merging_collision_direct = self._load_kernel("compute_merging_collision_direct", KERNEL_FILE_LOCATION)
    
        self._mass_intermediate = self.alloc(self.N, dtype=np.float32)
        self._radius_intermediate = self.alloc(self.N, dtype=np.float32)
        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)        
        self._position_intermediate = self.alloc(self.N, dtype=np.complex64)        
        self._flags_intermediate = self.alloc(self.N, dtype=np.uint32)        
        
        self._groups = np.arange(self.N, dtype=np.uint32) 
        self._groups_prev = np.arange(self.N, dtype=np.uint32)

    def compute_merging_collision_direct(self, state:PrimaryStateVectors, contacting:FindContactingBodiesPipeline, edge_distance:DistancePipeline):
        return self._compute_merging_collision_direct(
                self.queue,
                (self.N * config.MAX_NUM_CONTACTS_PER_BODY, ),
                (config.MAX_NUM_CONTACTS_PER_BODY, ),
                
                # Args
                np.uint32(self.N),
                state.flags,
                state.position,
                state.velocity,
                state.mass,
                state.radius,
                contacting.n_direct_contacts,
                contacting.direct_contacts,
                self._flags_intermediate,
                self._position_intermediate,
                self._velocity_intermediate,
                self._mass_intermediate,
                self._radius_intermediate
            )
    
    def __call__(self, state:PrimaryStateVectors, contacting:FindContactingBodiesPipeline, edge_distance:DistancePipeline):
        self.compute_merging_collision_direct(state, contacting, edge_distance)
        cl.enqueue_copy(self.queue, state.flags,    self._flags_intermediate)
        cl.enqueue_copy(self.queue, state.position, self._position_intermediate)
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)
        cl.enqueue_copy(self.queue, state.mass,     self._mass_intermediate)
        cl.enqueue_copy(self.queue, state.radius,   self._radius_intermediate)

    def find_merged_bodies(self):
        """Finds the merge events that happened since the last time called."""
        cl.enqueue_copy(self.queue,  self._groups, self._groups)
        diff = np.argwhere(self._groups != self._groups_prev)
        if len(diff) > 0: diff = diff[0]        
        merged_ids = np.column_stack((diff, self._groups[diff]))
        self._groups_prev[:] = self._groups
        return merged_ids