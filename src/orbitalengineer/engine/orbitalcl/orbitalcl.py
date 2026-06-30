from typing import cast
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import pyopencl as cl

from orbitalengineer.engine import log_timing, logger, config
from orbitalengineer.engine.event import BouncingCollisionEvent
from orbitalengineer.engine.metric import MetricsProducer
from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.device import CLDeviceManager
from orbitalengineer.engine.orbitalcl.dimension import load_kernel
from orbitalengineer.engine.orbitalcl.particle_cl import ParticleCL
from orbitalengineer.engine.orbitalcl.tracer import EventTracer

# Integrator kernels
from orbitalengineer.engine.orbitalcl.merge.merge import MergePipeline
from orbitalengineer.engine.orbitalcl.interaction.interaction import InteractionGroupPipeline
from orbitalengineer.engine.orbitalcl.nudge.nudge import NudgePipeline
from orbitalengineer.engine.orbitalcl.position.position import PositionPipeline
from orbitalengineer.engine.orbitalcl.relative_velocity.relative_velocity import RelativeVelocityPipeline
from orbitalengineer.engine.orbitalcl.velocity.velocity import VelocityPipeline
from orbitalengineer.engine.orbitalcl.bounce.bounce import BouncePipeline

import warnings
warnings.filterwarnings(
    "ignore",
    message="Non-empty compiler output encountered.*",
    module="pyopencl.cache"
)

mf = cl.mem_flags
kernel_dir = Path(__file__).parent

class SimController_CL:
    coef_of_restitution = config.COEF_OF_RESTITUTION
    dt_base = config.DEFAULT_DT_BASE
    G = config.DEFAULT_G
    EPS_DIST:float = config.EPS_DIST
    EPS_TIME:float = config.EPS_TIME
    
    N:int = 1024
    Lx:int = 256
    
    def __init__(self):
        self._active_events:list[cl.Event] = []
        self._particles:list = []
        self.accum = 0.0
        self.last_now:float|None = None
        self.enable_profiling = config.EMIT_METRICS
        self.is_initialized = False

        self.tr = EventTracer(self)
        self.metrics = MetricsProducer(config.METRIC_SOCKET_PATH)

        self.tick_id = 0
        self.step_count = 0
        self.device = None
        
        self.N = 1
        self._allocate_memory()

    @log_timing
    def _populate_particle_fields(self):
        for i,p in enumerate(self._particles):
            self.flags[i] = np.uint32(p.get_flags())
            self.velocity[i] = np.complex64(p.get_velocity())
            self.position[i] = np.complex64(p .get_position())
            self.mass[i] = np.float32(p.get_mass())
            self.radius[i] = np.float32(p.get_radius())

    @log_timing
    def _allocate_memory(self):
        self.flags = np.zeros(self.N, dtype=np.uint32)
        self.velocity = np.zeros(self.N, dtype=np.complex64)
        self.position = np.zeros(self.N, dtype=np.complex64)
        self.radius = np.zeros(self.N, dtype=np.float32)
        self.mass = np.zeros(self.N, dtype=np.float32)
        self.velocity_relative = np.zeros(self.N * self.N, dtype=np.complex64)
        self.distance_edge = np.zeros(self.N * self.N, dtype=np.float32)

    @log_timing
    def _create_buffers(self):
        self.flags_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flags)
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.pos_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.position)
        self.mass_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.mass)
        self.radius_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.radius)
        self.velocity_relative_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity_relative)
        self.distance_edge_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.distance_edge)

    @log_timing
    def _init_queue(self):
        properties = cast(cl.command_queue_properties, 0)
        if self.enable_profiling:
            logger.info("OpenCL profiling enabled.")
            properties = cl.command_queue_properties.PROFILING_ENABLE
        self.q = cl.CommandQueue(self.ctx, properties=properties)
    
    @log_timing
    def _generate_headers(self):
        build_dir = Path(".opencl_build")
        build_dir.mkdir(exist_ok=True)
        header_path = build_dir / "flags.clh"
        header_path.write_text(flags.generate_cl_flag_file())
        return build_dir
    
    def _get_build_contants(self):
        defs = {
            "COEF_OF_RESTITUTION": f'{self.coef_of_restitution}f',
            'G':                   f'{self.G}f',
            'EPS_DIST':            f'{self.EPS_DIST}f',
            'EPS_TIME':            f'{self.EPS_TIME}f',
        }
        return [ f'-D{k}={v}' for k,v in defs.items() ]
    
    @log_timing
    def _init_kernel(self):
        if self.device is None:
            #from warnings import warn
            #warn("Cannot initialize kernels until CL device is selected.", stacklevel=2)
            raise Exception("Cannot initialize kernels until CL device is selected.")

        self.ctx = cl.Context([self.device._device])
        self._init_queue()
        build_dir = self._generate_headers()                
        
        build_options = [
            '-cl-std=CL2.0',
            f'-I {kernel_dir}',
            f'-I {build_dir}',
            *self._get_build_contants()
        ]
        
        try:
            self._velocity = VelocityPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._position = PositionPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._bounce = BouncePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._interaction = InteractionGroupPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._nudge = NudgePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._merge = MergePipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self._relative_velocity = RelativeVelocityPipeline(self.N, self.ctx, self.q, self.tr, build_options)
            self.knl_distance_edge = load_kernel('compute_edge_distance', 'position/distance_edge.cl', self.ctx, build_options)
        except (cl._cl.RuntimeError, cl._cl.LogicError) as e: #type:ignore
            import sys
            print(e, file=sys.stderr)
            raise SystemExit
    
    def set_cl_device(self, platform_id:int, device_id:int):
        if self.is_initialized:
            logger.error("Cannot set CL device after initialization.")
            return
        self.device = CLDeviceManager(platform_id, device_id)
        self.ctx = cl.Context([self.device._device])        
        logger.info("CL device set to (%s, %s)", platform_id, device_id)
    
    @log_timing
    def init_sim(self):
        if self.is_initialized:
            logger.warning("Attempted to init_sim after simulation was already started.")
            return
        if len(self._particles):
            self.N = len(self._particles)
            self._allocate_memory()
            self._populate_particle_fields()
        self._init_kernel()
        self._create_buffers()
        self._interaction(self.dt_base, self.flags_cl, self.pos_cl, self.vel_cl, self.radius_cl, self.mass_cl)
        self._relative_velocity(self.pos_cl, self.vel_cl, self.velocity_relative_cl)
        self.tr.add("edge_distance", self.compute_distance_edge())
        #self._nudge(self.pos_cl, self.mass_cl, self.distance_edge_cl)
        self.is_initialized = True

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

    def _has_queue(self) -> bool:
        if not hasattr(self, 'q'):
            logger.warning("")
            return False
        return True

    def sync(self):
        if not self._has_queue(): return
        self.q.finish()
        cl.enqueue_copy(self.q, self.flags, self.flags_cl)
        cl.enqueue_copy(self.q, self.position, self.pos_cl)
        cl.enqueue_copy(self.q, self.velocity, self.vel_cl)
        cl.enqueue_copy(self.q, self.mass, self.mass_cl)
        cl.enqueue_copy(self.q, self.radius, self.radius_cl)
        cl.enqueue_copy(self.q, self._interaction.toi, self._interaction.toi_cl)
        self.q.finish()

    def sync_to_device(self):
        if not self._has_queue(): return
        self.q.finish()
        cl.enqueue_copy(self.q, self.flags_cl, self.flags)
        cl.enqueue_copy(self.q, self.pos_cl, self.position)
        cl.enqueue_copy(self.q, self.vel_cl, self.velocity)
        cl.enqueue_copy(self.q, self.mass_cl, self.mass)
        cl.enqueue_copy(self.q, self.radius_cl, self.radius)
        cl.enqueue_copy(self.q, self._interaction.toi_cl, self._interaction.toi)
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

    def drift(self, dt_step):
        self._position(dt_step, self.flags_cl, self.vel_cl, self.pos_cl)
        self.tr.add("edge_distance", self.compute_distance_edge())

    def sub_step(self, dt_step):
        self.kick(dt_step / 2.0)
        self.drift(dt_step)
        self.kick(dt_step / 2.0)

        self._relative_velocity(self.pos_cl, self.vel_cl, self.velocity_relative_cl)
        self._merge(dt_step, self.flags_cl, self.velocity_relative_cl, self.distance_edge_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl)
        self._bounce(self.flags_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl, self.velocity_relative_cl, self.distance_edge_cl)
        self._interaction(dt_step, self.flags_cl, self.pos_cl, self.vel_cl, self.radius_cl, self.mass_cl)
    
    def single_step(self, dt_step):
        dt_step_start = dt_step
        count = 0
        while dt_step > 0 and count < config.MAX_SUB_STEPS:
            cl.enqueue_copy(self.q, self._interaction.node_dt, self._interaction.node_dt_cl).wait()
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
        
        self._merge.find_merged_bodies()
        
        return num_steps

    def to_dict(self) -> dict:
        self.sync()
        return {
            # Simulation state
            "tick_id": int(self.tick_id),
            "dt_base": float(self.dt_base),
            "step_count": int(self.step_count),
            "Lx": int(self.Lx),
            "N": int(self.N),
            
            # Constants
            "G": float(self.G),
            "coef_of_restitution": float(self.coef_of_restitution),
            "EPS_DIST": float(self.EPS_DIST),
            "EPS_TIME": float(self.EPS_TIME),
            
            # particle field vectors
            "flags":    [int(fl) for fl in self.flags],
            "position": [(f"{p.real:.6f}", f"{p.imag:.6f}") for p in self.position],
            "velocity": [(f"{v.real:.6f}", f"{v .imag:.6f}") for v in self.velocity],
            "mass":     [f"{m:.6f}" for m in self.mass],
            "radius":   [f"{r:.6f}" for r in self.radius],
        }
    
    def load_from_dict(self, obj:dict):
        for field in ['tick_id', 'dt_base', 'step_count', 'Lx', 'N', 'G', 'EPS_DIST', 'EPS_TIME']:
            setattr(self, field, obj[field])
        
        def vector_complex64(values:list):
            return np.array([np.complex64(float(v[0]), float(v[1])) for v in values], dtype=np.complex64)

        def vector_float32(values:list):
            return np.array([np.float32(v) for v in values], dtype=np.float32)

        def vector_uint32(values:list):
            return np.array([np.uint32(v) for v in values], dtype=np.uint32)
        
        self.flags =vector_uint32(obj['flags'])
        self.position = vector_complex64(obj['position'])
        self.velocity = vector_complex64(obj['velocity'])
        self.mass = vector_float32(obj['mass'])
        self.radius = vector_float32(obj['radius'])
        self.velocity_relative = np.zeros(self.N * self.N, dtype=np.complex64)
        self.distance_edge = np.zeros(self.N * self.N, dtype=np.float32)
        return self