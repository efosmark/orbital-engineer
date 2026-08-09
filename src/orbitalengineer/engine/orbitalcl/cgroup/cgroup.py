import numpy as np
import pyopencl as cl
from orbitalengineer.engine.config import CGROUP_ASSIGN_MAX_ITERATIONS
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep


class CGroupAssignException(Exception):
    def __init__(self, num_iterations:int, num_bodies:int):
        self.num_iterations = num_iterations
        self.max_iterations = CGROUP_ASSIGN_MAX_ITERATIONS
        self.num_bodies = num_bodies
        
        super().__init__("\n".join([
            "Kernel cgroup_assign required too many iterations to complete.",
            "num_iterations={}, max_iterations={}, num_bodies={} ".format(
                self.num_iterations, self.max_iterations, self.num_bodies
            ),
            "This can be configured via the CGROUP_ASSIGN_MAX_ITERATIONS setting."
        ]))

KERNEL_FILE_LOCATION = "cgroup/cgroup.cl"

class CGroupPipeline(CLPipelineStep):
    debug_flag = "cgroup"

    def initialize(self):
        self._find_contacting_bodies = self._load_kernel("find_contacting_bodies", KERNEL_FILE_LOCATION)
        self._find_contacting_bodies_reduce = self._load_kernel("find_contacting_bodies_reduce", KERNEL_FILE_LOCATION)
        self._cgroup_assign = self._load_kernel("cgroup_assign", KERNEL_FILE_LOCATION)
        
        self.num_contacts_by_lane = np.zeros(self.N * self.N, dtype=np.uint32)
        self.num_contacts_by_lane_cl = self._create_buffer(self.num_contacts_by_lane)

        self.contacts_by_lane = np.zeros(self.N * self.N, dtype=np.uint32)
        self.contacts_by_lane_cl = self._create_buffer(self.contacts_by_lane)
        
        self.num_contacts = np.zeros(self.N, dtype=np.uint32)
        self.num_contacts_cl = self._create_buffer(self.num_contacts)
        
        self.contacts_reduced = np.zeros(self.N * self.N, dtype=np.uint32)
        self.contacts_reduced_cl = self._create_buffer(self.contacts_reduced)
        
        self.has_updates = np.zeros(self.N, dtype=np.uint32)
        self.has_updates_cl = self._create_buffer(self.has_updates)
        
        self.edge_dist = np.zeros(self.N * self.N, dtype=np.float32)
        self.edge_dist_cl = self._create_buffer(self.edge_dist)
     
    def _print_collisions_per_body(self):
        """Debug printing of the collision matrix."""
        cl.enqueue_copy(self.queue, self.contacts_reduced, self.contacts_reduced_cl).wait()
        cl.enqueue_copy(self.queue, self.num_contacts, self.num_contacts_cl).wait()
        print()
        print()
        for i in range(self.N):
            try:
                if self.num_contacts[i] == 0:
                    continue
                print(f" {i:3.0f} [{self.num_contacts[i]:2.0f}] | ", end="")
                contacts = [
                    f"{self.contacts_reduced[(self.N * i) + j]:3.0f}"
                    for j in range(self.num_contacts[i])
                ]
                print(" ".join(contacts))
            except IndexError:
                break

    def find_contacting_bodies(self, flags: cl.Buffer, position: cl.Buffer, radius: cl.Buffer, time_of_interaction: cl.Buffer):        
        self.tr.add("find_contacting_bodies",
            self._find_contacting_bodies(
                self.queue,
                (self.N * self.N, ),  # global work size
                (self.Lx,   ),        # local work size
                
                # Args
                np.uint32(self.N),
                flags,
                position,
                radius,
                time_of_interaction,
                self.num_contacts_by_lane_cl,
                self.contacts_by_lane_cl
            )
        )
        
        cl.enqueue_copy(self.queue, self.num_contacts_by_lane, self.num_contacts_by_lane_cl).wait()
        self.tr.add("find_contacting_bodies_reduce",
            self._find_contacting_bodies_reduce(
                self.queue,
                (self.N, ),  # global work size
                (1, ),        # local work size
                
                # Args
                np.uint32(self.N),
                np.uint32(self.Lx),
                self.num_contacts_by_lane_cl,
                self.contacts_by_lane_cl,
                self.num_contacts_cl,
                self.contacts_reduced_cl
            )
        )

    def cgroup_assign(self, cgroups: cl.Buffer):
        cl.enqueue_copy(self.queue, self.num_contacts, self.num_contacts_cl).wait()
        all_colliding_ids = np.array([
            i for i in range(self.num_contacts.size) 
            if self.num_contacts[i] > 0
        ], dtype=np.uint32)        
        if all_colliding_ids.size == 0: return
        all_colliding_ids_cl = self._create_buffer(all_colliding_ids)
        
        local_size = int(np.max(self.num_contacts))
        
        cgroups_B = np.arange(self.N, dtype=np.uint32)
        cgroups_B_cl = self._create_buffer(cgroups_B)

        cl.enqueue_copy(self.queue, cgroups, np.arange(self.N, dtype=np.uint32)).wait()
        if local_size <= 0 or all_colliding_ids.size <= 0:
            return
        
        updated = True
        num_iterations = 0
        while updated:
            updated = False
            self.has_updates[:] = 0
            cl.enqueue_copy(self.queue, self.has_updates_cl, self.has_updates).wait()
            
            self.tr.add("cgroup_assign",
                self._cgroup_assign(
                    self.queue,
                    (all_colliding_ids.size * local_size, ),  # global work size
                    (local_size, ),                           # local work size
                    
                    # Args
                    np.uint32(self.N),
                    all_colliding_ids_cl,
                    self.num_contacts_cl,
                    self.contacts_reduced_cl,
                    cgroups,
                    cgroups_B_cl,
                    self.has_updates_cl
                )
            )

            cl.enqueue_copy(self.queue, self.has_updates, self.has_updates_cl).wait()
            updated = self.has_updates.any()
            num_iterations += 1
            
            cl.enqueue_copy(self.queue, cgroups, cgroups_B_cl).wait()
                        
            if num_iterations >= self.N or num_iterations >= CGROUP_ASSIGN_MAX_ITERATIONS:
                raise CGroupAssignException(num_iterations, self.N)
    
    def __call__(self, flags: cl.Buffer, position: cl.Buffer, radius: cl.Buffer, time_of_interaction: cl.Buffer, cgroups_buffer: cl.Buffer):
        self.find_contacting_bodies(flags, position, radius, time_of_interaction)        
        
        new_cgroups = np.arange(self.N, dtype=np.uint32)
        new_cgroups_buffer = self._create_buffer(new_cgroups)
        self.cgroup_assign(new_cgroups_buffer)
        cl.enqueue_copy(self.queue, cgroups_buffer, new_cgroups_buffer).wait()
    