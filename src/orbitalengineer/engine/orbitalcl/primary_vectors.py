import os
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import pyopencl as cl

mf = cl.mem_flags
DEFAULT_MEM_FLAGS = mf.READ_WRITE|mf.COPY_HOST_PTR

from orbitalengineer import flags
from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalcl.named_shared_memory import NamedSharedMemory
from orbitalengineer.engine.orbitalcl.tracer import EventTracer
from orbitalengineer.helpers import r_from_mass
from orbitalengineer.ipc import message

DEFAULT_LOCAL_SIZE = 256

class PrimaryStateVectors:
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
    
    def __init__(self, shm:NamedSharedMemory, ctx:cl.Context, queue:cl.CommandQueue, tr:EventTracer):
        self.shm = shm
        self.ctx = ctx
        self.queue = queue
        self.tr = tr
        self._host_vector:dict[cl.Buffer, NDArray] = dict()
    
    def initialize(self, N):
        self.N_bodies_alloc = N
        self.N_bodies_valid = N
        self.create_default_sizes(N)
        self._allocate_primary_vectors()
    
    def alloc(self, size:int, dtype:type|np.dtype, fill:Sequence|None=None, shared_name:None|str=None, mem_flags=DEFAULT_MEM_FLAGS) -> cl.Buffer:
        if shared_name is not None:
            vec = self.shm.create_shared_memory(shared_name, size, dtype)
        elif fill is not None:
            vec = np.array(fill, dtype=dtype)
        else:
            vec = np.zeros(size, dtype)
        b = self._create_buffer(vec, mem_flags=mem_flags)
        self._host_vector[b] = vec
        return b
    
    def get_host_vector(self, buffer: cl.Buffer, sync:bool=False) -> NDArray:
        vec = self._host_vector[buffer]
        if sync:
            self.sync_to_host(buffer).wait()
        return vec
    
    def sync_to_host(self, buffer: cl.Buffer, queue:cl.CommandQueue|None=None) -> cl.Event:
        if queue is None:
            queue = self.queue
        return self.tr.add('enqueue_copy', cl.enqueue_copy(queue, self.get_host_vector(buffer), buffer))
    
    def sync_to_device(self, buffer: cl.Buffer) -> cl.Event:
        return self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, buffer, self.get_host_vector(buffer)))
    
    def _create_buffer(self, hostbuf, mem_flags=DEFAULT_MEM_FLAGS) -> cl.Buffer:
        return cl.Buffer(self.ctx, mem_flags, hostbuf=hostbuf)
    
    def update_num_bodies(self):
        flags_host = self.get_host_vector(self.flags, sync=True)
        self.N_bodies_alloc = int(flags_host.size)
        self.N_bodies_valid = int(np.count_nonzero(flags_host[(flags_host&flags.REMOVED) == 0]))
        self.create_default_sizes(self.N_bodies_valid)
    
    def create_default_sizes(self, N:int):
        self.Lx = DEFAULT_LOCAL_SIZE
        self.N = N
        if self.N < self.Lx:
            self.Lx = int(self.N)
        self.grid_stride_global_size = (self.N_bodies_valid * self.Lx, )
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

    def _allocate_primary_vectors(self):
        self.id_to_index = self.alloc(self.N, np.uint32, shared_name='id_to_index')
        self.body_id = self.alloc(self.N, np.uint32, shared_name='body_id')
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
    
    def defragment(self):
        return
        with self.tr("defragment (host)"):
            
            # Copy all of the state vectors to the host 
            id_to_index_host = self.get_host_vector(self.id_to_index, sync=True)
            body_id_host = self.get_host_vector(self.body_id, sync=True)
            flags_host = self.get_host_vector(self.flags, sync=True)
            position_host = self.get_host_vector(self.position, sync=True)
            velocity_host = self.get_host_vector(self.velocity, sync=True)
            mass_host = self.get_host_vector(self.mass, sync=True)
            radius_host = self.get_host_vector(self.radius, sync=True)
            
            # Get the indices for sorting the removed bodies to the end
            
            idx = np.arange(self.N_bodies_valid)
            idx_removed = np.argwhere((flags_host[:self.N_bodies_valid]&flags.REMOVED)).flatten()
            idx_end = np.arange(start=idx.size - idx_removed.size, stop=idx.size)
            
            idx[idx_removed] = idx_end
            idx[idx_end] = idx_removed
            #print(idx_removed, idx_end)
            #return
            
            #idx = np.argsort((flags_host&flags.REMOVED), stable=True)
            
            # Replace with sorted vectors
            id_to_index_host[idx] = np.arange(idx.size)#
            body_id_host = body_id_host[idx]
            flags_host = flags_host[idx]
            position_host = position_host[idx]
            velocity_host = velocity_host[idx]
            mass_host = mass_host[idx]
            radius_host = radius_host[idx]
            
            #print(id_to_index_host)
            
            # Push the sorted values back to the device
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.id_to_index, id_to_index_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.body_id, body_id_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.flags, flags_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.position, position_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.velocity, velocity_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.mass, mass_host))
            self.tr.add('enqueue_copy', cl.enqueue_copy(self.queue, self.radius, radius_host))
            self.queue.finish()

        self.update_num_bodies()