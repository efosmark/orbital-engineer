import pyopencl as cl
import numpy as np
from orbitalengineer.engine.dimension import PipelineComponent
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors

mf = cl.mem_flags

DEFAULT_MEM_FLAGS = mf.READ_WRITE|mf.COPY_HOST_PTR

KERNEL_FILE_LOCATION = "energy/energy.cl"

class EnergyPipeline(PipelineComponent):
    debug_flag = 'energy'
    
    _evt_compute_energy:cl.Event|None = None
    
    def initialize(self):
        self._knl_compute_energy = self._load_kernel("compute_energy", KERNEL_FILE_LOCATION)
        self.kinetic_energy = self.vec.alloc(self.vec.N, dtype=np.float64, shared_name="kinetic_energy")
        self.grav_potential_energy = self.vec.alloc(self.vec.N, dtype=np.float64, shared_name="grav_potential_energy")

    def __call__(self, state:PrimaryStateVectors):
        state.sync(self.queue)
        self._evt_compute_energy = self._knl_compute_energy(
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
                self.kinetic_energy,
                self.grav_potential_energy
        ).wait()