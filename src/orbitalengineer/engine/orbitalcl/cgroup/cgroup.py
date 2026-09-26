import numpy as np
import pyopencl as cl
from orbitalengineer.engine import logger, config
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline

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
        self._n_updated = self.alloc(1, dtype=np.uint32)
        
        self._default_groups = self.alloc(self.N, dtype=np.uint32, fill=list(np.arange(self.N, dtype=np.uint32)))
        self.group = self.alloc(self.N, np.uint32, shared_name="cgroup")
        self.new_cgroups_A = self.alloc(self.N, dtype=np.uint32, fill=list(np.arange(self.N, dtype=np.uint32)))
        self.new_cgroups_B = self.alloc(self.N, dtype=np.uint32, fill=list(np.arange(self.N, dtype=np.uint32)))

    def cgroup_assign(self, dist:DistancePipeline):
        all_colliding_ids, global_size, local_size = dist.get_all_colliding_ids()
        if global_size == 0:
            return
        
        cl.enqueue_copy(self.queue, self.new_cgroups_A, self._default_groups)
        
        num_iterations = 0
        updated = True
        while updated:
            try:
                self._cgroup_assign(
                    self.queue,
                    (global_size, ),
                    (local_size,  ),
                    
                    # Args
                    np.uint32(self.N),
                    all_colliding_ids,
                    dist.n_nearby_contacts,
                    dist.nearby_contacts,
                    self.new_cgroups_A,
                    self.new_cgroups_B,
                    self._n_updated
                )
            except cl.LogicError as e:
                logger.error("cgroup_assign failed: %s", e)
                logger.error("global_size=%s local_size=%s", global_size, local_size)
                break

            with self.tr("(cpu) cgroup_get_updated"):
                n_updated = self.get_host_vector(self._n_updated)
                cl.enqueue_copy(self.queue, n_updated[:1], self._n_updated).wait()
                if n_updated[0] == 0: break

            # Sync the most up-to-date results back to the source buffer
            self.tr.add("cgroup_sync", cl.enqueue_copy(self.queue, self.new_cgroups_A, self.new_cgroups_B)) #.wait() 
            
            num_iterations += 1
            if num_iterations >= self.N or num_iterations >= config.CGROUP_ASSIGN_MAX_ITERATIONS:
                raise CGroupAssignException(self.N)

        self.tr.add("cgroup_sync", cl.enqueue_copy(self.queue, self.group, self.new_cgroups_B)) #.wait()   
    
    def __call__(self, dist:DistancePipeline):
        self.cgroup_assign(dist)