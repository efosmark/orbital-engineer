from typing import Sequence

import numpy as np
import pyopencl as cl

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.helpers import r_from_mass
from orbitalengineer.ipc import message

DEFAULT_LOCAL_SIZE = 256

class PrimaryStateVectors(PipelineComponent):
    initialized:bool = False
    
    N_bodies_alloc:int
    N_bodies_valid:int
    N:int
    Lx:int
    
    body_id:cl.Buffer
    id_to_index:cl.Buffer
    flags:cl.Buffer
    velocity:cl.Buffer
    position:cl.Buffer
    mass:cl.Buffer
    radius:cl.Buffer
    
    def initialize(self):
        self.N_bodies_alloc = self.N
        self.N_bodies_valid = self.N
        self.create_default_sizes(self.N)
        self.allocate_memory()
    
    def set_N_valid(self, N_bodies_valid):
        self.N_bodies_valid = N_bodies_valid
        self.create_default_sizes(N_bodies_valid)
    
    def create_default_sizes(self, N:int):
        self.Lx = DEFAULT_LOCAL_SIZE
        self.N = N
        if self.N < self.Lx:
            self.Lx = int(self.N)
        self.grid_stride_global_size = (self.N * self.Lx, )
        self.grid_stride_local_size = (self.Lx, )
    
    def populate(self, particles:Sequence[message.ParticleInit]):
        if self.initialized:
            logger.warning("PrimaryStateVectors was already initialized and populated. Recreating.")
                
        id_host = self.get_host_vector(self.body_id)
        id_to_index_host = self.get_host_vector(self.id_to_index)
        flags_host = self.get_host_vector(self.flags)
        velocity = self.get_host_vector(self.velocity)
        position = self.get_host_vector(self.position)
        mass = self.get_host_vector(self.mass)
        radius = self.get_host_vector(self.radius)
        
        for i,p in enumerate(particles):
            id_host[i] = np.uint32(i)
            id_to_index_host[i] = np.uint32(i)
            flags_host[i] = np.uint32(p.flags)
            velocity[i] = np.complex64(*p.velocity)
            position[i] = np.complex64(*p.position)
            mass[i] = np.float32(p.mass)
            radius[i] = np.float32(p.radius)

        cl.enqueue_copy(self.queue, self.body_id, id_host)
        cl.enqueue_copy(self.queue, self.flags, flags_host)
        cl.enqueue_copy(self.queue, self.velocity, velocity)
        cl.enqueue_copy(self.queue, self.position, position)
        cl.enqueue_copy(self.queue, self.mass, mass)
        cl.enqueue_copy(self.queue, self.radius, radius)
        
        self.initialized = True
        logger.info("Populated %s bodies", len(particles))

    def allocate_memory(self):
        self.body_id = self.alloc(self.N, np.uint32, shared_name='body_id')
        self.id_to_index = self.alloc(self.N, np.uint32, shared_name='id_to_index')
        self.flags = self.alloc(self.N, np.uint32, shared_name='flags')
        self.velocity = self.alloc(self.N, np.complex64, shared_name='velocity')
        self.position = self.alloc(self.N, np.complex64, shared_name='position')
        self.mass = self.alloc(self.N, np.float32, shared_name='mass')
        self.radius = self.alloc(self.N, np.float32, shared_name='radius')
    
    def apply_vector_offset(self, vector_name:str, ids:Sequence[int], op:str, offset:tuple[float,float]):
        if vector_name == "position":
            buffer = self.position
            value = np.complex64(offset[0], offset[1])
        elif vector_name == "velocity":
            buffer = self.velocity
            value = np.complex64(offset[0], offset[1])
        elif vector_name == "mass":
            buffer = self.mass
            value = np.float32(offset[0])
        elif vector_name == "radius":
            buffer = self.radius
            value = np.float32(offset[0])
        else:
            logger.error("Invalid vector name for apply_vector_offset. "
                         "Must be one of: position, velocity, mass, or radius.")
            return False
        
        vector = self.get_host_vector(buffer)
        if op == 'add':
            vector[ids] += value
        elif op == 'mul':
            vector[ids] *= value
        cl.enqueue_copy(self.queue, buffer, vector)
        
        if vector_name == "mass":
            radius = self.get_host_vector(self.radius)
            mass = self.get_host_vector(self.mass)
            radius[ids] = np.vectorize(r_from_mass)(mass[ids])
            cl.enqueue_copy(self.queue, self.radius, mass) 
        
        return True

    def sync(self, queue):
        self.sync_to_host(self.id_to_index, queue=queue)
        self.sync_to_host(self.body_id, queue=queue)
        self.sync_to_host(self.flags, queue=queue)
        self.sync_to_host(self.velocity, queue=queue)
        self.sync_to_host(self.position, queue=queue)
        self.sync_to_host(self.mass, queue=queue)
        self.sync_to_host(self.radius, queue=queue)
    
    
    # def defrag(self):
    #     with self.tr('defrag (host)'):
    #         self.sync(self.queue)
            
    #         flags_host = self.get_host_vector(self.flags, sync=True)
    #         filter |= (flags_host & flags.REMOVED) == flags.REMOVED   # Only valid bodies

    #         # zero out for any invalid bodies (ensures they never get chosen as the focus)
    #         force[filter] = 0.0
            
    #         for i in range(self.N_bodies_valid):
    #             if (self.flags & flags.REMOVED) == flags.REMOVED