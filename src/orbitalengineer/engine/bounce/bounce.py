import numpy as np
import pyopencl as cl
from orbitalengineer import dtypes
from orbitalengineer.engine.cgroup.cgroup import CGroupPipeline
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.distance.distance import DistancePipeline
from orbitalengineer.engine.interaction.interaction import InteractionPipeline
from orbitalengineer.engine.ledger.ledger import LedgerController
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors
from orbitalengineer.engine.velocity_along_normal.velocity_along_normal import VelocityAlongNormalPipeline

KERNEL_FILE_LOCATION = "bounce/bounce.cl"

class BouncePipeline(PipelineComponent):
    debug_flag = "bounce"
    
    def initialize(self):
        self._collide_bounce_simple = self._load_kernel("compute_bouncing_collision_simple", KERNEL_FILE_LOCATION)
        self._collide_bounce_single = self._load_kernel("compute_bouncing_collision_single", KERNEL_FILE_LOCATION)

        self._velocity_intermediate = self.vec.alloc(self.vec.N, dtype=dtypes.COMPLEX)

    def collide_bounce_single(self,  state:PrimaryStateVectors, interact:InteractionPipeline, ledger:LedgerController):
        self._collide_bounce_single(
            self.queue,
            (self.vec.N_bodies_valid * self.vec.Lx, ),
            (self.vec.Lx, ),
                        
            # Args
            np.uint32(self.vec.N_bodies_alloc),
            np.uint32(self.vec.N_bodies_valid),
            state.flags,
            state.position,
            state.velocity,
            state.mass,
            state.radius,
            interact.dt_until_collision,
            self._velocity_intermediate,
            ledger.ledger_entry_count,
            ledger.ledger
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)

    def collide_bounce_simple(self, state:PrimaryStateVectors, interact:InteractionPipeline, distance: DistancePipeline, velocity_along_normal: VelocityAlongNormalPipeline):
        cl.enqueue_copy(self.queue, self._velocity_intermediate, state.velocity)
        self._collide_bounce_simple(
            self.queue,
            (self.vec.N_bodies_valid * self.vec.Lx, ),
            (self.vec.Lx, ),
                        
            # Args
            np.uint32(self.vec.N_bodies_alloc),
            np.uint32(self.vec.N_bodies_valid),
            state.flags,
            state.position,
            state.velocity,
            state.mass,
            state.radius,
            interact.dt_until_collision,
            distance.edge_to_edge,
            velocity_along_normal.velocity_along_normal,
            self._velocity_intermediate
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate).wait()

    def __call__(self, state:PrimaryStateVectors, interact:InteractionPipeline, cgroup:CGroupPipeline, distance: DistancePipeline, velocity_along_normal: VelocityAlongNormalPipeline, ledger:LedgerController):
        cl.enqueue_copy(self.queue, self._velocity_intermediate, np.zeros(state.N, dtype=dtypes.COMPLEX))
        self.collide_bounce_single(state, interact, ledger)        
