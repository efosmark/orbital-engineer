from typing import cast
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import pyopencl as cl

from orbitalengineer.engine import logger, config
from orbitalengineer.engine.cl import flags
from orbitalengineer.engine.cl.interaction import InteractionGroupPipeline
from orbitalengineer.engine.cl.nudge import NudgePipeline
from orbitalengineer.engine.cl.position import PositionPipeline
from orbitalengineer.engine.cl.relative_velocity import RelativeVelocityPipeline
from orbitalengineer.engine.cl.velocity import VelocityPipeline
from orbitalengineer.engine.cl.bounce import BouncePipeline
from orbitalengineer.engine.cl.dimension import load_kernel
from orbitalengineer.engine.cl.merge import MergePipeline
from orbitalengineer.engine.cl.particle_cl import ParticleCL
from orbitalengineer.engine.cl.tracer import EventTracer
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.event import BouncingCollisionEvent
from orbitalengineer.engine.metric import MetricsProducer
from orbitalengineer.engine.simcontroller import OrbitalSimController
from orbitalengineer.engine.cl.device import get_device_pci_bdf, map_pci_bdf_to_drm_card

PLATFORM_ID = 0
DEVICE_ID = 1

# particle_dtype = np.dtype([
#     ("velocity", np.complex64),
#     ("position", np.complex64),
#     ("mass",     np.float32),
#     ("radius",   np.float32)
# ])

import warnings
warnings.filterwarnings(
    "ignore",
    message="Non-empty compiler output encountered.*",
    module="pyopencl.cache"
)

mf = cl.mem_flags
kernel_dir = Path(__file__).parent

def round_up(n, l):
    return ((n + l - 1) // l) * l

class SimController_CL(OrbitalSimController):
    collision_strategy = config.DEFAULT_COLLISION_STRATEGY
    coef_of_restitution = config.COEF_OF_RESTITUTION
    dt_base = config.DEFAULT_DT_BASE
    G = config.DEFAULT_G

    N:int = 1024
    Lx:int = 256
    
    def __init__(self, clock:SimClock):
        self._active_events:list[cl.Event] = []
        self._particles:list = []
        self.clock = clock
        self.accum = 0.0
        self.last_now:float|None = None
        self.enable_profiling = True
        self.is_initialized = False

        self.tr = EventTracer(self)
        self.metrics = MetricsProducer(config.METRIC_SOCKET_PATH)

        self.step_count = 0
        self.opencl_device_name = ""
        self.opencl_platform_name = ""
        self.opencl_pci_bdf:str|None = None
        self.drm_card_index:int|None = None

    def _populate_particle_fields(self):
        for i,p in enumerate(self._particles):
            self.flags[i] = np.uint32(p.get_flags())
            self.velocity[i] = np.complex64(p.get_velocity())
            self.position[i] = np.complex64(p .get_position())
            self.mass[i] = np.float32(p.get_mass())
            self.radius[i] = np.float32(p.get_radius())

    def _allocate_memory(self):
        # Primary particle fields
        self.flags = np.zeros(self.N, dtype=np.uint32)
        self.velocity = np.zeros(self.N, dtype=np.complex64)
        self.position = np.zeros(self.N, dtype=np.complex64)
        self.radius = np.zeros(self.N, dtype=np.float32)
        self.mass = np.zeros(self.N, dtype=np.float32)
        
        self.velocity_relative = np.zeros(self.N * self.N, dtype=np.complex64)
        self.distance_edge = np.zeros(self.N * self.N, dtype=np.float32)
        logger.debug("Memory allocated.")

    def _create_buffers(self):
        self.flags_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flags)
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.pos_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.position)
        self.mass_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.mass)
        self.radius_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.radius)
        self.velocity_relative_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity_relative)
        self.distance_edge_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.distance_edge)
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
            logger.info("OpenCL profiling enabled.")
            properties = cl.command_queue_properties.PROFILING_ENABLE
        self.copy_q = cl.CommandQueue(self.ctx, properties=properties)
        self.q = cl.CommandQueue(self.ctx, properties=properties)
    
    def _generate_headers(self):
        build_dir = Path(".opencl_build")
        build_dir.mkdir(exist_ok=True)
        header_path = build_dir / "flags.clh"
        header_path.write_text(flags.generate_cl_flag_file())
        return build_dir
    
    def _init_kernel(self):
        self._init_device()
        self.ctx = cl.Context([self._device])
        self._init_queue()
        build_dir = self._generate_headers()                
        
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
            f'-I {build_dir}',
            *[ f'-D{k}={v}' for k,v in defs.items() ]
        ]
        
        try:
            self._velocity = VelocityPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._position = PositionPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._bounce = BouncePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._interaction = InteractionGroupPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._nudge = NudgePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._merge = MergePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            
            self._relative_velocity = RelativeVelocityPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self.knl_distance_edge = load_kernel('compute_edge_distance', 'kernel/distance_edge.cl', self.ctx, build_options)
            
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
        self.compute_distance_edge()
        
        self.is_initialized = True
        logger.info("Sim was initialized.")

    def add_particle(self, particle) -> int:
        if self.is_initialized:
            logger.warning("Cannot add a particle once the simulation has started.")
            return -1
        self._particles.append(particle)
        return len(self._particles) - 1
    
    def set_position(self, body_id, x, y):
        self.position[body_id] = complex(x, y)
        cl.enqueue_copy(self.q, self.pos_cl, self.position).wait()
    
    def set_velocity(self, body_id, x, y):
        self.velocity[body_id] = complex(x, y)
        cl.enqueue_copy(self.q, self.vel_cl, self.velocity).wait()

    def find_bodies_at(self, x:float, y:float, margin:float=10):
        indices = self.get_valid_indices()
        
        # Apply a bit of margin to the radius (e.g. if a radius is too small, it cant be clicked)
        radius = self.radius[indices] + margin

        # Relative difference between the click and every location
        p = self.position[indices]
        d = np.complex128(x, y) - p
        
        # Create a mask indicating where there is crossover
        mask = (np.abs(d) <= radius)
        
        # Get the indices 
        return indices[mask]

    def get_valid_indices(self) -> NDArray:
        """Get the list of nodes that are still considered valid (e.g. not removed)."""
        return np.where((self.flags & flags.REMOVED) != flags.REMOVED)[0]

    def get_particle(self, particle_id:int):
        return ParticleCL(particle_id, self)

    def __iter__(self):
        for i in self.get_valid_indices():
            yield self.get_particle(int(i))

    def sync(self):
        self.q.finish()
        cl.enqueue_copy(self.q, self.flags, self.flags_cl)
        cl.enqueue_copy(self.q, self.position, self.pos_cl)
        cl.enqueue_copy(self.q, self.velocity, self.vel_cl)
        cl.enqueue_copy(self.q, self.mass, self.mass_cl)
        cl.enqueue_copy(self.q, self.radius, self.radius_cl)
        cl.enqueue_copy(self.q, self._interaction.toi, self._interaction.toi_cl)
        self.q.finish()
    
    def compute_distance_edge(self):
        return self.knl_distance_edge(
            self.q,
            (self.N * self.Lx, ),  # global work size
            (self.Lx, ),           # local work size
            
            # Args
            np.uint32(self.N),
            self.pos_cl,
            self.radius_cl,
            self.distance_edge_cl
        )
    
    def kick(self, dt_step):
        self._velocity(dt_step, self.flags_cl, self.pos_cl, self.mass_cl, self.radius_cl, self.distance_edge_cl, self.vel_cl)

    def kick2(self, dt_step):
        self._velocity(dt_step, self.flags_cl, self.pos_cl, self.mass_cl, self.radius_cl, self.distance_edge_cl, self.vel_cl)
        self._relative_velocity(self.pos_cl, self.vel_cl, self.velocity_relative_cl)
        self._merge(dt_step, self.flags_cl, self.velocity_relative_cl, self.distance_edge_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl)
        self._bounce(self.flags_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl, self.velocity_relative_cl, self.distance_edge_cl)

    def drift(self, dt_step):
        self._position(dt_step, self.flags_cl, self.vel_cl, self.pos_cl)
        self.tr.add("edge_distance", self.compute_distance_edge())

    def sub_step(self, dt_step):
        self.kick(dt_step / 2.0)
        self.drift(dt_step)
        self.kick2(dt_step / 2.0)
        self._interaction(dt_step, self.flags_cl, self.pos_cl, self.vel_cl, self.radius_cl, self.mass_cl)
    
    def single_step(self, dt_step):
        dt_step_start = dt_step
        count = 0
        
        while dt_step > 0 and count < config.MAX_SUB_STEPS:
            cl.enqueue_copy(self.copy_q, self._interaction.node_dt, self._interaction.node_dt_cl).wait()
            
            try:
                min_toi = min([t for t in self._interaction.node_dt if t > 0])
            except ValueError:
                min_toi = dt_step_start / config.MAX_SUB_STEPS
            
            dt = max(dt_step_start / config.MAX_SUB_STEPS, min(dt_step, min_toi))
            self.sub_step(dt)
            dt_step -= dt
            self.step_count += 1
            count += 1
                
        return count, dt_step_start - dt_step

    def _detect_event(self):
        cl.enqueue_copy(self.q, self.distance_edge, self.distance_edge_cl)
        cl.enqueue_copy(self.q, self._bounce.collision_point, self._bounce.collision_point_cl)
        cl.enqueue_copy(self.q, self.velocity_relative, self.velocity_relative_cl)
        
        for i in range(self.N):
            row_offset = i * self.N
            for j in range(i + 1, self.N):
                grid_idx = row_offset + j
                edge_dist = self.distance_edge[grid_idx]
                point = self._bounce.collision_point[grid_idx]
                if point != 0:
                    evt = BouncingCollisionEvent(self.tick_id, i, j, complex(point), complex(self.velocity_relative[grid_idx]), float(edge_dist))
                    self.on_collision(evt)
    
    def on_collision(self, event:BouncingCollisionEvent):...

    def tick(self, now):
        if not self.is_initialized:
           logger.warning("tick() was called before simulation initialization.")
           return 0
        
        if self.last_now is None:
            self.accum = 0
            self.last_now = now - self.dt_base
 
        wall_dt = now - self.last_now
        self.accum += wall_dt

        dt_step = np.float32(self.dt_base) # <- for now, fixed cap
        num_steps = int(min(self.accum // dt_step, config.MAX_STEPS_PER_TICK)) # type:ignore
        self.last_now = now
        
        if num_steps > 0:
            count, dt_processed = self.single_step(dt_step)
        
            #self.accum -= dt_processed # type: ignore
            self.accum -= dt_step # type: ignore
            
            if config.EMIT_METRICS and self.enable_profiling:
                self.q.finish()
                timeline = self.tr.timeline()
                self.tr.clear()

                self.metrics.emit_metric("tick", self.tick_id, timeline=[
                    { "name":t["name"], "duration_ms":round(t["duration_ns"]/1e6, 6)}
                    for t in timeline
                ], dt_step=round(float(dt_step), 6))
            
            self.tick_id += 1
        
        return num_steps
