import numpy as np
import pyopencl as cl
from orbitalengineer.engine import logger, config
from orbitalengineer.engine.orbitalcl.contacting.contacting import FindContactingBodiesPipeline
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent


class CGroupAssignException(Exception):
    def __init__(self, num_bodies:int):
        self.max_iterations = config.CGROUP_ASSIGN_MAX_ITERATIONS
        self.num_bodies = num_bodies
        
        super().__init__("\n".join([
            "Kernel cgroup_assign required too many iterations to complete.",
            "num_iterations={}, num_bodies={} ".format(
                self.max_iterations, self.num_bodies
            ),
            "This can be configured via the CGROUP_ASSIGN_MAX_ITERATIONS setting."
        ]))

KERNEL_FILE_LOCATION = "cgroup/cgroup.cl"

class CGroupPipeline(PipelineComponent):
    debug_flag = "cgroup"

    def initialize(self):
        self._cgroup_assign = self._load_kernel("cgroup_assign", KERNEL_FILE_LOCATION)
        self._num_contacts_host = np.zeros(self.N, dtype=np.uint32)
        self._has_updates = np.zeros(self.N, dtype=np.uint32)
        self._has_updates_cl = self._create_buffer(self._has_updates)

        self._cgroups_host = self.shm.create_shared_memory('cgroup', self.N, np.uint32)
        self.cgroups = self._create_buffer(self._cgroups_host)

    def cgroup_assign(self, contacting: FindContactingBodiesPipeline, new_cgroups: cl.Buffer):
        all_colliding_ids, global_size, local_size = contacting.get_ids()

        
        cgroups_B = np.arange(self.N, dtype=np.uint32)
        cgroups_B_cl = self._create_buffer(cgroups_B)

        cl.enqueue_copy(self.queue, new_cgroups, np.arange(self.N, dtype=np.uint32)).wait()
        
        updated = True
        num_iterations = 0
        while updated:
            updated = False
            #self._has_updates[:] = 0
            #cl.enqueue_copy(self.queue, self._has_updates_cl, self._has_updates).wait()
            
            try:
                self._cgroup_assign(
                    self.queue,
                    (global_size, ),
                    (local_size,  ),
                    
                    # Args
                    np.uint32(self.N),
                    all_colliding_ids,
                    contacting.num_contacts_cl,
                    contacting.contacts_cl,
                    new_cgroups,
                    cgroups_B_cl,
                    self._has_updates_cl
                )
            except cl.LogicError as e:
                logger.error("cgroup_assign failed: %s", e)
                logger.error("global_size=%s local_size=%s", global_size, local_size)
                print(all_colliding_ids)
                break

            cl.enqueue_copy(self.queue, self._has_updates, self._has_updates_cl).wait()
            updated = self._has_updates.any()
            num_iterations += 1
            
            cl.enqueue_copy(self.queue, new_cgroups, cgroups_B_cl).wait()    
            if num_iterations >= self.N or num_iterations >= config.CGROUP_ASSIGN_MAX_ITERATIONS:
                raise CGroupAssignException(self.N)
        
        print(f"{num_iterations=}")
    
    def __call__(self, contacting: FindContactingBodiesPipeline, cgroups_buffer: cl.Buffer):        
        new_cgroups = np.arange(self.N, dtype=np.uint32)
        new_cgroups_buffer = self._create_buffer(new_cgroups)
        self.cgroup_assign(contacting, new_cgroups_buffer)
        cl.enqueue_copy(self.queue, cgroups_buffer, new_cgroups_buffer).wait()
    