import numpy as np
import pyopencl as cl

from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

KERNEL_FILE_LOCATION = "defragment/defragment.cl"

class DefragBodies(PipelineComponent):
    debug_flag = "defrag"

    def initialize(self):
        self._defragment_removed_bodies = self._load_kernel("defrag_orbital_vectors", KERNEL_FILE_LOCATION)

        self._N_max = self.alloc(1, dtype=np.uint32, fill=[0])
        
        self._id_to_index_intermediate = self.alloc(self.N, dtype=np.uint32)      
        self._body_id_intermediate = self.alloc(self.N, dtype=np.uint32)      
        self._flags_intermediate = self.alloc(self.N, dtype=np.uint32)      
        self._position_intermediate = self.alloc(self.N, dtype=np.complex64)        
        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)        
        self._mass_intermediate = self.alloc(self.N, dtype=np.float32)
        self._radius_intermediate = self.alloc(self.N, dtype=np.float32)
    
    def __call__(self, state:PrimaryStateVectors):
        N_max = int(self.get_host_vector(self._N_max, sync=True)[0])
        
        evt = self._defragment_removed_bodies(
                self.queue,
                (int(self.N - N_max), ),
                (64, ),
                
                # Args
                np.uint32(self.N),
                self._N_max,
                
                state.id_to_index,
                state.body_id,
                state.flags,
                state.position,
                state.velocity,
                state.mass,
                state.radius,
                
                self._id_to_index_intermediate,
                self._body_id_intermediate,
                self._flags_intermediate,
                self._position_intermediate,
                self._velocity_intermediate,
                self._mass_intermediate,
                self._radius_intermediate,
            )
        
        evt.wait()
        
        #print("N = ", self.N, flush=True)
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.id_to_index, self._id_to_index_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.body_id, self._body_id_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.flags, self._flags_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.position, self._position_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.mass, self._mass_intermediate))
        self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, state.radius, self._radius_intermediate))
        self.queue.finish()
        
        return evt
