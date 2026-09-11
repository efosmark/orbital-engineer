import numpy as np
import pyopencl as cl
from pyopencl import array
from orbitalengineer.engine.orbitalcl.cgroup.cgroup import CGroupPipeline
from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.interaction.interaction import InteractionPipeline
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors
from orbitalengineer.engine.orbitalcl.velocity_along_normal.velocity_along_normal import VelocityAlongNormalPipeline

KERNEL_FILE_LOCATION = "bounce/bounce.cl"

class BouncePipeline(PipelineComponent):
    debug_flag = "bounce"
    
    def initialize(self):
        self._collide_bounce_simple = self._load_kernel("compute_bouncing_collision_simple", KERNEL_FILE_LOCATION)
        self._collide_bounce_single = self._load_kernel("compute_bouncing_collision_single", KERNEL_FILE_LOCATION)
        self._compute_center_of_mass = self._load_kernel("compute_center_of_mass", KERNEL_FILE_LOCATION)
        self._compute_impulse = self._load_kernel("compute_impulse", KERNEL_FILE_LOCATION)
        self._assign_impulse = self._load_kernel("assign_impulse", KERNEL_FILE_LOCATION)

        self._velocity_intermediate = self.alloc(self.N, dtype=np.complex64)

        self._impulse = array.zeros(self.queue, self.N * self.N, dtype=np.complex64)

    def compute_impulse(self, state:PrimaryStateVectors, contacting: FindContactingBodiesPipeline, cgroup: cl.Buffer):
        # Clear out the impulse table
        self._impulse.fill(np.complex64(0, 0))

        ids, global_size, local_size = contacting.get_ids(state.flags)        
        if ids is None:
            return

        com_momentum = np.zeros(self.N, dtype=np.complex64)
        com_velocity = np.zeros(self.N, dtype=np.complex64)
        com_total_mass = np.zeros(self.N, dtype=np.float32)
        
        com_momentum_cl = self._create_buffer(com_momentum)
        com_velocity_cl = self._create_buffer(com_velocity)
        com_total_mass_cl = self._create_buffer(com_total_mass)
    
        self._compute_center_of_mass(
            self.queue,
            (global_size, ),  # global work size
            (local_size, ),                           # local work size
            
            # Args
            np.uint32(self.N),
            state.flags,
            state.position,
            state.velocity,
            state.mass,
            ids,
            contacting.n_direct_contacts,
            contacting.direct_contacts,
            com_momentum_cl,
            com_velocity_cl,
            com_total_mass_cl
        )
        
        self._compute_impulse(
            self.queue,
            (global_size, ),  # global work size
            (local_size, ),                           # local work size
            
            # Args
            np.uint32(self.N),
            state.flags,
            state.position,
            state.velocity,
            state.mass,
            ids,
            contacting.n_direct_contacts,
            contacting.direct_contacts,
            #total_momentum_cl,
            com_momentum_cl,
            com_velocity_cl,
            com_total_mass_cl,
            self._impulse.data
        )

        cl.enqueue_copy(self.queue, self._velocity_intermediate, state.velocity)
        
        if self._impulse.data is None:
            return
        
        self._assign_impulse(
            self.queue,
            (global_size, ),  # global work size
            (local_size, ),                           # local work size
            
            # Args
            np.uint32(self.N),
            ids,
            contacting.n_direct_contacts,
            contacting.direct_contacts,
            self._impulse.data,
            self._velocity_intermediate
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)

    def collide_bounce_single(self,  state:PrimaryStateVectors, interact:InteractionPipeline):
        self._collide_bounce_single(
            self.queue,
            state.grid_stride_global_size,
            state.grid_stride_local_size,
                        
            # Args
            np.uint32(self.N),
            state.flags,
            state.position,
            state.velocity,
            state.mass,
            state.radius,
            interact.dt_until_collision,
            self._velocity_intermediate
        )
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)

    def collide_bounce_simple(self, state:PrimaryStateVectors, interact:InteractionPipeline, distance: DistancePipeline, velocity_along_normal: VelocityAlongNormalPipeline):
        cl.enqueue_copy(self.queue, self._velocity_intermediate, state.velocity)
        self._collide_bounce_simple(
            self.queue,
            state.grid_stride_global_size,
            state.grid_stride_local_size,
                        
            # Args
            np.uint32(self.N),
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
        cl.enqueue_copy(self.queue, state.velocity, self._velocity_intermediate)


    def __call__(self, state:PrimaryStateVectors, interact:InteractionPipeline, contacting: FindContactingBodiesPipeline, cgroup:CGroupPipeline, distance: DistancePipeline, velocity_along_normal: VelocityAlongNormalPipeline):
        cl.enqueue_copy(self.queue, self._velocity_intermediate, np.zeros(self.N, dtype=np.complex64))
        self.collide_bounce_single(state, interact)
        #self.collide_bounce_simple(state, interact, distance, velocity_along_normal)
        #self.compute_impulse(flags, position, velocity, mass, contacting, cgroup)
        
