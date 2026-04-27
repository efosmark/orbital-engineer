from dataclasses import dataclass
from typing import Any, ClassVar, Sequence, cast

from orbitalengineer.engine import logger, config
from orbitalengineer.engine.cl.particle_cl import ParticleCL
from orbitalengineer.engine.cl.old.substep import SubStep
from orbitalengineer.engine.simcontroller import OrbitalSimController, Particle
from orbitalengineer.engine.cl.device import get_device_pci_bdf, map_pci_bdf_to_drm_card

from orbitalengineer.engine.cl.kernelizer import EventTracer, load_kernel
from orbitalengineer.engine.cl.drift import Drift
from orbitalengineer.engine.cl.global_reduce_min import GlobalReduceMin
from orbitalengineer.engine.cl.impact import ComputeImpact
from orbitalengineer.engine.cl.kick import KickPairwise
from orbitalengineer.engine.cl.old.nudge import NudgeOverlaps
from orbitalengineer.engine.cl.pairwise_node_reduce import PairwiseNodeReduce
from orbitalengineer.engine.cl.pairwise_min_reduce import PairwiseMinReduce

import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#os.environ['PYOPENCL_CACHE'] = '0'

PLATFORM_ID = 1
DEVICE_ID = 0

import warnings
warnings.filterwarnings(
    "ignore",
    message="Non-empty compiler output encountered.*",
    module="pyopencl.cache"
)

import numpy as np
import pyopencl as cl
mf = cl.mem_flags

from pathlib import Path
kernel_dir = Path(__file__).parent


IN_NONE = 0
IN_APPROACHING = 1
IN_COLLISION = 2
IN_DEPARTING = 3
IN_ORBITING = 4
IN_ORBITED = 5


def round_up(n, l):
    return ((n + l - 1) // l) * l

@dataclass
class Metric:
    field:str
    value:float|int

@dataclass
class TickStats:
    num_steps:int
    tick_id:int
    metrics:list[Metric]

class SimController_CL(OrbitalSimController):
    speed = config.DEFAULT_SPEED
    dt_base = config.DEFAULT_DT_BASE
    coef_of_restitution = config.COEF_OF_RESTITUTION
    G = config.DEFAULT_G

    N:int = 1024
    Lx:int = 256
    
    def __init__(self):
        self._active_events:list[cl.Event] = []
        self._particles:list = []
        self.accum = 0.0
        self.last_now:float|None = None
        self.enable_profiling = True
        self.is_initialized = False
        self.tr = EventTracer(self)

        self.step_count = 0
        self.opencl_device_name = ""
        self.opencl_platform_name = ""
        self.opencl_pci_bdf:str|None = None
        self.drm_card_index:int|None = None

    def _populate_particle_fields(self):
        for i,p in enumerate(self._particles):
            self.velocity[i] = np.complex64(p.get_velocity())
            self.position[i] = np.complex64(p.get_position())
            self.mass[i] = np.float32(p.get_mass())
            self.radius[i] = np.float32(p.get_radius())

    def _allocate_memory(self):
        self.velocity = np.zeros(self.N, dtype=np.complex64)
        self.velocity_relative = np.zeros(self.N * self.N, dtype=np.complex64)
        self.position = np.zeros(self.N, dtype=np.complex64)
        self.radius = np.zeros(self.N, dtype=np.float32)
        self.mass = np.zeros(self.N, dtype=np.float32)
        self.distance_edge = np.zeros(self.N * self.N, dtype=np.float32)
        self.time_to_impact = np.array([np.inf for i in range(self.N)], dtype=np.float32)
        self.acceleration = np.zeros(self.N * self.N, dtype=np.complex64)


        self.toi = np.zeros(self.N * self.N, dtype=np.float32)
        #self.interactions = np.array([0 for i in range(self.pairs_host.size)], dtype=np.float32)

        logger.debug("Memory allocated.")

    def _create_buffers(self):
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.pos_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.position)
        self.mass_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.mass)        
        self.radius_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.radius)
        self.time_to_impact_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.time_to_impact)
        self.acceleration_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.acceleration)
        self.velocity_relative_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity_relative)
        self.distance_edge_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.distance_edge)

        #self.pair_vel_ij_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.pair_vel_ij)
        #self.pair_vel_ji_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.pair_vel_ji)
        
        #self.pair_pos_ij_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.pair_pos_ij)
        #self.pair_pos_ji_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.pair_pos_ji)
        
        self.toi_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.toi)
        #self.interactions_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.interactions)
        logger.debug("Buffers were created.")
    
    def _init_device(self):
        platform = cl.get_platforms()[PLATFORM_ID]
        devices = platform.get_devices()
        self._device = devices[-1]
        self.opencl_platform_name = platform.name
        self.opencl_device_name = self._device.name
        self.opencl_pci_bdf = get_device_pci_bdf(self._device)
        self.drm_card_index = map_pci_bdf_to_drm_card(self.opencl_pci_bdf)

    def _init_queue(self):
        properties = cast(cl.command_queue_properties, 0)
        if self.enable_profiling:
            properties = cl.command_queue_properties.PROFILING_ENABLE
        self.copy_q = cl.CommandQueue(self.ctx, properties=properties)
        self.q = cl.CommandQueue(self.ctx, properties=properties)
    
    def _init_kernel(self):
        self._init_device()
        self.ctx = cl.Context([self._device])
        self._init_queue()
        
        defs = {
            "COEF_OF_RESTITUTION": f'{self.coef_of_restitution}f',
            'G':        f'{self.G}f',
            'EPS_DIST': f'{config.EPS_DIST}f',
            'EPS_TIME': f'{config.EPS_TIME}f',
            'DV_MIN':   f'{config.DV_MIN}f',
            'DV_MAX':   f'{config.DV_MAX}f'
        }
        
        build_options = [
            '-cl-std=CL2.0',
            f'-I {kernel_dir}',
            *[ f'-D{k}={v}' for k,v in defs.items() ]
        ]
        
        try:
            self.knl_accel = load_kernel('compute_acceleration', 'kernel/acceleration.cl', self.ctx, build_options)
            self.knl_position = load_kernel('compute_position', 'kernel/position.cl', self.ctx, build_options)
            self.knl_interaction = load_kernel('compute_interactions', 'kernel/interaction.cl', self.ctx, build_options)
            self.knl_velocity = load_kernel('compute_velocity', 'kernel/velocity.cl', self.ctx, build_options)
            self.knl_velocity_relative = load_kernel('compute_velocity_relative', 'kernel/velocity_relative.cl', self.ctx, build_options)
            self.knl_collide = load_kernel('compute_collision', 'kernel/collide.cl', self.ctx, build_options)
            self.knl_distance_edge = load_kernel('compute_distance_edge', 'kernel/distance_edge.cl', self.ctx, build_options)
            #self.kick_pairwise = KickPairwise(self.ctx, self.q, self.tr, build_options)
            #self.drift_direct = Drift(self.ctx, self.q, self.tr, build_options)
            #self.nudge_overlaps = NudgeOverlaps(self.ctx, self.q, self.tr, build_options)
            logger.info("Kernels have been created.")
        except (cl._cl.RuntimeError, cl._cl.LogicError) as e: #type:ignore
            import sys
            print(e, file=sys.stderr)
            raise SystemExit
    
    def init_sim(self):
        if self.is_initialized:
            logger.warning("Attempted to init_sim after simulation was already started.")
            return
        self.N = len(self._particles)
        self.N_alloc = round_up(self.N, 256)
        self._init_kernel()
        self._allocate_memory()
        self._populate_particle_fields()
        self._create_buffers()
        
        self.is_initialized = True
        logger.info("Sim was initialized.")

    def add_particle(self, particle) -> int:
        if self.is_initialized:
            logger.warning("Cannot add a particle once the simulation has started.")
            return -1
        self._particles.append(particle)
        return len(self._particles) - 1
    
    def find_bodies_at(self, x:float, y:float, margin:float=10):
        indices = np.arange(self.N)
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.radius[indices] + margin

        # Relative difference between the click and every location
        p = self.position[indices]
        d = np.complex128(x, y) - p
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius)
        
        # Get the indices 
        return indices[mask]

    def get_valid_indices(self):
        return np.arange(self.N)

    def get_particle(self, particle_id:int):
        return ParticleCL(particle_id, self)

    def __iter__(self):
        for i in range(self.N):
            yield self.get_particle(int(i))

    def sync(self):
        self.q.finish()
        cl.enqueue_copy(self.q, self.position, self.pos_cl)
        cl.enqueue_copy(self.q, self.velocity, self.vel_cl)
        cl.enqueue_copy(self.q, self.toi, self.toi_cl)
        self.q.finish()

    def compute_acceleration(self, dt_step):
        Lx = 256
        return self.knl_accel(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            np.float32(dt_step),
            self.pos_cl,
            self.mass_cl,
            self.radius_cl,
            self.acceleration_cl
        )

    def compute_interaction(self):
        Lx = 256
        return self.knl_interaction(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            self.pos_cl,
            self.vel_cl,
            self.radius_cl,
            self.toi_cl,
            self.time_to_impact_cl#,
            #self.interactions_cl
        )
    
    def compute_position(self, dt_step):
        return self.knl_position(
            self.q,
            (self.N,),   # global work size
            None,        # local work size
                        
            # Args
            np.float32(dt_step),
            self.vel_cl,
            self.pos_cl
        )
    
    def compute_relative_velocity(self):
        Lx = 256
        return self.knl_velocity_relative(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
                        
            # Args
            np.uint32(self.N),
            self.pos_cl,
            self.vel_cl,
            self.radius_cl,
            self.velocity_relative_cl
        )
        
    def compute_velocity(self, dt_step):
        Lx = 256
        return self.knl_velocity(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
                        
            # Args
            np.uint32(self.N),
            np.float32(dt_step),
            self.acceleration_cl,
            self.pos_cl,
            self.vel_cl,
            self.mass_cl,
            self.distance_edge_cl
        )
    
    def compute_distance_edge(self):
        Lx = 256
        return self.knl_distance_edge(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            self.pos_cl,
            self.mass_cl,
            self.radius_cl,
            self.distance_edge_cl
        )
    
    def compute_collision(self):
        Lx = 256
        return self.knl_collide(
            self.q,
            (self.N * Lx, ),  # global work size
            (Lx, ),           # local work size
                        
            # Args
            np.uint32(self.N),
            self.velocity_relative_cl,
            self.pos_cl,
            self.vel_cl,
            self.mass_cl,
            self.distance_edge_cl
        )

    def kick(self, dt_step):
        self.tr.add("acceleration", self.compute_acceleration(dt_step))
        self.tr.add("velocity", self.compute_velocity(dt_step))
        self.tr.add("v_rel", self.compute_relative_velocity())
        self.tr.add("collision", self.compute_collision())

    def drift(self, dt_step):
        self.tr.add("position", self.compute_position(dt_step))
        self.tr.add("distance_edge", self.compute_distance_edge())

    def sub_step(self, dt_step):
        self.kick(dt_step / 2.0)
        self.drift(dt_step)
        self.kick(dt_step / 2.0)
        
        self.tr.add("impact_time", self.compute_interaction())
    
    def single_step(self, dt_step):
        dt_step_start = dt_step
        count = 0
        while dt_step > 0 and count < 12:
            
            cl.enqueue_copy(
                self.copy_q,
                self.time_to_impact,
                self.time_to_impact_cl
            ).wait()
            
            try:
                min_toi = min([t for t in self.time_to_impact if t > 0])
            except ValueError:
                min_toi = config.EPS_TIME
            
            dt = max(config.EPS_TIME, min(dt_step, min_toi))
            self.sub_step(dt)
            dt_step -= dt
            self.step_count += 1
            count += 1
            #break
                
        return dt_step_start - dt_step

    def tick(self, now):
        if not self.is_initialized:
           logger.warning("tick() was called before simulation initialization.")
           return 0
            
        if self.last_now is None:
            self.accum = 0
            self.last_now = now - self.dt_base
 
        wall_dt = now - self.last_now
        self.accum += wall_dt * self.speed

        dt_step = np.float32(self.dt_base) # <- for now, fixed cap

        num_steps = int(min(self.accum // dt_step, config.MAX_STEPS_PER_FRAME)) # type:ignore
        self.last_now = now
        
        if num_steps > 0:
            dt_processed = self.single_step(dt_step)
            self.tick_id += 1
        
            self.q.finish()
            timeline = self.tr.timeline()
            self.tr.clear()

            #self.accum -= dt_processed # type: ignore
            self.accum -= dt_step # type: ignore
            
            return TickStats(
                num_steps,
                self.tick_id,
                [
                    *[ Metric(t["name"], t["duration_ns"]/1000.0/1000.0) for t in timeline],
                    # Metric("tick", (time.monotonic() - tick_start) * 1000.0)
                ] if self.enable_profiling else []
            )
        
        return TickStats(0, self.tick_id, [])
