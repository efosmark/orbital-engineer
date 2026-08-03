import numpy as np
import pyopencl as cl
from orbitalengineer.engine.orbitalcl.dimension import CLPipelineStep
from orbitalengineer.engine import logger

KERNEL_FILE_LOCATION = "cgroup/cgroup.cl"

class CGroupPipeline(CLPipelineStep):
    debug_flag = "cgroup"

    def initialize(self):
        #self._compute_edge_distance = self._load_kernel("compute_edge_distance", KERNEL_FILE_LOCATION)
        #self._collision_group_assign = self._load_kernel("collision_group_assign", KERNEL_FILE_LOCATION)
        
        
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
        self._n_iter_max = 5
    
    # def compute_edge_distance(self, position_buffer: cl.Buffer, radius_buffer: cl.Buffer):
    #     self.tr.add("collision_group_assign",
    #         self._compute_edge_distance(
    #             self.queue,
    #             (self.N * self.Lx, ),  # global work size
    #             (self.Lx, ),           # local work size
                
    #             # Args
    #             np.uint32(self.N),
    #             position_buffer,
    #             radius_buffer,
    #             self.edge_dist_cl
    #         )
    #     )
    
    def find_contacting_bodies(self, flags: cl.Buffer, position_buffer: cl.Buffer, radius_buffer: cl.Buffer):        
        self.tr.add("find_contacting_bodies",
            self._find_contacting_bodies(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                flags,
                position_buffer,
                radius_buffer,
                self.edge_dist_cl,
                self.num_contacts_by_lane_cl,
                self.contacts_by_lane_cl
            )
        )

        self.tr.add("find_contacting_bodies_reduce",
            self._find_contacting_bodies_reduce(
                self.queue,
                (self.N, ),  # global work size
                None,        # local work size
                
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


        all_colliding_ids = self.get_all_colliding_ids()
        
        cgroups_B = np.arange(self.N, dtype=np.uint32)
        cgroups_B_cl = self._create_buffer(cgroups_B)

        cl.enqueue_copy(self.queue, cgroups, np.arange(self.N, dtype=np.int32)).wait()
        

        
        updated = True
        num_iterations = 0
        while updated and num_iterations < 1000:
            #print()
            #print(f'---[{num_iterations}]---------------------------')
            updated = False
            self.has_updates[:] = 0
            cl.enqueue_copy(self.queue, self.has_updates_cl, self.has_updates).wait()
                        
            
            self.tr.add("cgroup_assign",
                self._cgroup_assign(
                    self.queue,
                    (all_colliding_ids.size, ),  # global work size
                    (1, ),           # local work size
                    
                    # Args
                    np.uint32(self.N),
                    all_colliding_ids,
                    self.num_contacts_cl,
                    self.contacts_reduced_cl,
                    cgroups,
                    cgroups_B_cl,
                    self.has_updates_cl
                )
            )

            #cl.enqueue_copy(self.queue, cgroups, new_cgroups_buffer).wait()
            cl.enqueue_copy(self.queue, self.has_updates, self.has_updates_cl).wait()
            #print(f"{self.has_updates=}")
            updated = self.has_updates.any()
            num_iterations += 1
            
            
            cl.enqueue_copy(self.queue, cgroups, cgroups_B_cl).wait()
            
            #print('B', cgroups_B)
            
            #cgroups_A[:] = cgroups_B[:]
            #cl.enqueue_copy(self.queue, cgroups_A_cl, cgroups_A).wait()
            
            if num_iterations >= 10:
                raise Exception("TOO MANY ITERATIONS")

    
    def get_all_colliding_ids(self):
        cl.enqueue_copy(self.queue, self.num_contacts, self.num_contacts_cl).wait()
        ids_reduced = np.array([
            i for i in range(self.num_contacts.size) 
            if self.num_contacts[i] > 0
        ], dtype=np.uint32)
        ids_reduced_cl = self._create_buffer(ids_reduced)
        return ids_reduced_cl
    
    # def collision_group_assign(self, flags: cl.Buffer, radius_buffer: cl.Buffer, cgroups: cl.Buffer):
    #     cl.enqueue_copy(self.queue, cgroups, np.arange(self.N, dtype=np.int32)).wait()
        
    #     result_indices = np.zeros(self.N, dtype=np.uint32)
    #     result_buffer = self._create_buffer(result_indices)
        
    #     new_cgroups = np.arange(self.N, dtype=np.uint32)
    #     new_cgroups_buffer = self._create_buffer(new_cgroups)

    #     has_updates = True
    #     num_iterations = 0
    #     while has_updates:
    #         self.tr.add("collision_group_assign",
    #             self._collision_group_assign(
    #                 self.queue,
    #                 (self.N * self.Lx, ),  # global work size
    #                 (self.Lx, ),           # local work size
                    
    #                 # Args
    #                 np.uint32(self.N),
    #                 flags,
    #                 radius_buffer,
    #                 self.edge_dist_cl,
    #                 cgroups,
    #                 new_cgroups_buffer,
    #                 result_buffer
    #             )
    #         )

    #         cl.enqueue_copy(self.queue, cgroups, new_cgroups_buffer).wait()
    #         cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
    #         has_updates = result_indices.any()
    #         num_iterations += 1
        
    #     if num_iterations > self._n_iter_max:
    #         logger.warning("collision_group_assign n_iterations=%s", num_iterations)
    #         self._n_iter_max = num_iterations
    
    
    def __call__(self, flags: cl.Buffer, position_buffer: cl.Buffer, radius_buffer: cl.Buffer, cgroups_buffer: cl.Buffer):
        #self.compute_edge_distance(position_buffer, radius_buffer)
        self.find_contacting_bodies(flags, position_buffer, radius_buffer)
        
        
        #print(ids_reduced)
        #print(num_contacts)
        #print(np.column_stack((num_contacts, contacts_reduced.reshape((self.N, self.N)))))

        #cgroups_A = np.arange(self.N, dtype=np.uint32)
        #cgroups_A_cl = self._create_buffer(cgroups_A)
        
        #self.cgroup_assign(ids_reduced_cl, cgroups_A_cl)

        #cl.enqueue_copy(self.queue, cgroups_A, cgroups_A_cl).wait()
        #print('A', cgroups_A)
        
        
        
        new_cgroups = np.arange(self.N, dtype=np.uint32)
        new_cgroups_buffer = self._create_buffer(new_cgroups)
        self.cgroup_assign(new_cgroups_buffer)

        cl.enqueue_copy(self.queue, cgroups_buffer, new_cgroups_buffer).wait()
        
        # Temporary just to display values in console
        #cgroups_host = np.arange(self.N, dtype=np.uint32)
        #cl.enqueue_copy(self.queue, cgroups_host, new_cgroups_buffer).wait()
        #print(cgroups_host)
        
        #new_cgroups = np.zeros(self.N, dtype=np.uint32)
        #new_cgroups_buffer = self._create_buffer(new_cgroups)
        #cl.enqueue_copy(self.queue, new_cgroups_buffer, cgroups_buffer).wait()
        #self.collision_group_assign(flags, radius_buffer, new_cgroups_buffer)