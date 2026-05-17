import numpy as np
import pyopencl as cl
from orbitalengineer.engine.cl.dimension import CLPipelineStep
from orbitalengineer.engine.collision_strategy import CollisionStrategy

KERNEL_FILE_LOCATION = "kernel/interaction.cl"

class InteractionGroupPipeline(CLPipelineStep):
    debug_flag = 'interaction'
    
    def initialize(self):
        self._compute_interaction_time = self._load_kernel("compute_interaction_time", KERNEL_FILE_LOCATION)
        self._assign_interaction_groups = self._load_kernel("assign_interaction_groups", KERNEL_FILE_LOCATION)
        self._reduce_interaction_groups = self._load_kernel("reduce_interaction_groups", KERNEL_FILE_LOCATION)
        self._collect_group_members = self._load_kernel("collect_group_members", KERNEL_FILE_LOCATION)
    
        self.toi = np.zeros(self.N * self.N, dtype=np.float32)
        self.toi_cl = self._create_buffer(self.toi)

        self.node_dt = np.array([np.inf for i in range(self.N)], dtype=np.float32)
        self.node_dt_cl = self._create_buffer(self.node_dt)
        
        self.group_dt = np.array([np.inf for i in range(self.N)], dtype=np.float32)
        self.group_dt_cl = self._create_buffer(self.group_dt)
        
        self.groups = np.zeros(self.N, dtype=np.uint32)
        self.groups_cl = self._create_buffer(self.groups)    
    
    def compute_interaction_time(self, position: cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer):
        return self.tr.add("interaction_time",
            self._compute_interaction_time(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                position,
                velocity,
                radius,
                self.toi_cl,
                self.node_dt_cl
            )
        )
    
    def assign_interaction_groups(self, dt_step: float, mass: cl.Buffer):
        return self.tr.add("assign_interaction_groups",
            self._assign_interaction_groups(
                self.queue,
                (self.N * self.Lx, ),  # global work size
                (self.Lx, ),           # local work size
                
                # Args
                np.uint32(self.N),
                np.float32(dt_step),
                mass,
                self.toi_cl,
                self.groups_cl
            )
        )
    
    def reduce_interaction_groups(self, mass: cl.Buffer):
        num_workgroups = (self.N // self.Lx) + 1
        result_indices = np.zeros(num_workgroups, dtype=np.uint32)
        result_buffer = self._create_buffer(result_indices)
        has_updates = True
        
        n_iterations = 0
        while has_updates:
            n_iterations += 1
            self.tr.add("reduce_interaction_groups",
                self._reduce_interaction_groups(
                    self.queue,
                    (self.N, ),   # global work size
                    (self.Lx, ),  # local work size
                    
                    # Args
                    np.uint32(self.N),
                    mass,
                    self.node_dt_cl,
                    self.group_dt_cl,
                    self.groups_cl,
                    result_buffer,
                ))

            cl.enqueue_copy(self.queue, result_indices, result_buffer).wait()
            has_updates = result_indices.any()
            if n_iterations > 5: raise SystemExit

    # def collect_group_members(self ):
    #     return self.tr.add("collect_group_members",
    #         self._collect_group_members(
    #             self.queue,
    #             (self.N * self.Lx, ),  # global work size
    #             None, #(self.Lx, ),           # local work size
                
    #             # Args
    #             np.uint32(self.N),
    #             self.groups_cl,
    #             self.group_members_cl
    #         ))

    def __call__(self, dt_step:float, position:cl.Buffer, velocity: cl.Buffer, radius: cl.Buffer, mass: cl.Buffer):
        self.compute_interaction_time(position, velocity, radius)
        self.assign_interaction_groups(dt_step, mass)
        self.reduce_interaction_groups(mass)

        #cl.enqueue_copy(self.queue, self.node_dt, self.node_dt_cl).wait()
        #print("self.node_dt", self.node_dt)
        
        cl.enqueue_copy(self.queue, self.group_dt, self.group_dt_cl).wait()
        #print("self.group_dt", self.group_dt)


if __name__ == "__main__":
    from cmath import pi, rect
    import numpy as np
    
    from orbitalengineer.engine.collision_strategy import CollisionStrategy
    from orbitalengineer.engine.cl.orbitalcl import SimController_CL
    from orbitalengineer.engine.clock import SimClock
    from orbitalengineer.engine.particle import ParticleRaw
    
    N = 4
    clock = SimClock()
    ctl = SimController_CL(clock)

    for p_offset in [10000, -10000]:
        for i in range(N):
            phi = (i/float(N)) * 2.0 * pi
            ctl.add_particle(ParticleRaw(
                position=p_offset + rect( 100.0, phi),
                velocity=rect(-100.0, phi),
                radius=20,
                mass=10
            ))

    ctl.add_particle(ParticleRaw(position=1000+1000j, velocity=20-10j, radius=10, mass=10))
    ctl.add_particle(ParticleRaw(position=1500-1000j, velocity=-20-10j, radius=10, mass=10))
    ctl.add_particle(ParticleRaw(position=-1700-1000j, velocity=11+5j, radius=10, mass=10))

    ctl.Lx = int((N * 2.0) + 3)
    ctl.speed = 1.0
    
    ctl.collision_strategy = CollisionStrategy.BOUNCE
    ctl.coef_of_restitution = 0.8
    ctl.init_sim()
    
    print()
    print()
    print()
    
    
    now = clock.time()
    ctl.dt_base = 1/15
    for _ in range(11):
        print()
        print('-' * 80)
        
        s = ctl.tick(now)
        if s == 0: continue
        
        # cl.enqueue_copy(ctl.q, ctl._interaction.node_dt, ctl._interaction.node_dt_cl).wait()
        # print(' NODE DT', ctl._interaction.node_dt)
        
        # cl.enqueue_copy(ctl.q, ctl._interaction.group_dt, ctl._interaction.group_dt_cl).wait()
        # print('GROUP DT', ctl._interaction.group_dt)
        
        # cl.enqueue_copy(ctl.q, ctl._interaction.groups, ctl._interaction.groups_cl).wait()
        # print('GROUP ID', ctl._interaction.groups)
        
        #cl.enqueue_copy(ctl.q, ctl.position, ctl.pos_cl).wait()
        #print('P', [ f'{p:.2f}' for p in ctl.position ])
        
        #cl.enqueue_copy(ctl.q, ctl.velocity, ctl.vel_cl).wait()
        #print('V', [ f'{p:.2f}' for p in ctl.velocity ])

        # cl.enqueue_copy(ctl.q, ctl._interaction.group_members, ctl._interaction.group_members_cl).wait()
        # for row in ctl._interaction.group_members.reshape((ctl.N,ctl.N)):
        #     print("[", end="")
        #     for cell in row:
        #         if cell == -1:
        #             print(' -- ', end='')
        #         else:
        #             print(f" {cell:02.0f} ", end='')
        #     print("]")
        
        
        #print()
        #print(f"TICK:{s.tick_id}   N_STEPS:{s.num_steps}   DT:{s.dt_step:.3f}")
        now += ctl.dt_base
