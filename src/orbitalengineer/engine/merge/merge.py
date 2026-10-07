import numpy as np
import pyopencl as cl
from orbitalengineer.engine import config
from orbitalengineer.engine.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.distance.distance import DistancePipeline
from orbitalengineer.engine.ledger.ledger import LedgerController
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "merge/merge.cl"

class MergePipeline(PipelineComponent):
    debug_flag = "merge"

    def initialize(self):
        self._compute_merging_collision_direct = self._load_kernel("compute_merging_collision_direct", KERNEL_FILE_LOCATION)
    
        self._mass_intermediate = self.vec.alloc(self.vec.N, dtype=np.float32)
        self._radius_intermediate = self.vec.alloc(self.vec.N, dtype=np.float32)
        self._velocity_intermediate = self.vec.alloc(self.vec.N, dtype=np.complex64)        
        self._position_intermediate = self.vec.alloc(self.vec.N, dtype=np.complex64)        
        self._flags_intermediate = self.vec.alloc(self.vec.N, dtype=np.uint32)        
        
        self._groups = np.arange(self.vec.N, dtype=np.uint32) 
        self._groups_prev = np.arange(self.vec.N, dtype=np.uint32)

    def compute_merging_collision_direct(self, state:PrimaryStateVectors, distance:DistancePipeline, ledger:LedgerController):
        #print('merge', self.vec.N_bodies_alloc, self.vec.N_bodies_valid)
        return self._compute_merging_collision_direct(
                self.queue,
                (self.vec.N_bodies_alloc * config.MAX_NUM_CONTACTS_PER_BODY, ),
                (config.MAX_NUM_CONTACTS_PER_BODY, ),
                
                # Args
                np.uint32(self.vec.N_bodies_alloc),
                np.uint32(self.vec.N_bodies_valid),
                state.flags,
                state.position,
                state.velocity,
                state.mass,
                state.radius,
                distance.n_direct_contacts,
                distance.direct_contacts,
                self._flags_intermediate,
                self._position_intermediate,
                self._velocity_intermediate,
                self._mass_intermediate,
                self._radius_intermediate,
                ledger.ledger_entry_count,
                ledger.ledger,
            )
    
    def __call__(self, state:PrimaryStateVectors, distance:DistancePipeline, ledger:LedgerController):
        self.compute_merging_collision_direct(state, distance, ledger)
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.flags,    self._flags_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.position, self._position_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.mass,     self._mass_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.radius,   self._radius_intermediate))