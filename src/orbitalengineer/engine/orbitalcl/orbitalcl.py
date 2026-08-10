from multiprocessing import shared_memory
from typing import Sequence, cast
from pathlib import Path
from numpy.typing import NDArray

import numpy as np
import pyopencl as cl

from orbitalengineer.engine import log_timing, logger, config
from orbitalengineer.engine.exception import InitKernelException
from orbitalengineer.engine.metric import MetricsProducer

from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.cgroup.cgroup import CGroupPipeline
from orbitalengineer.engine.orbitalcl.device import CLDeviceManager
from orbitalengineer.engine.orbitalcl.tracer import EventTracer
from orbitalengineer.engine.orbitalcl.merge.merge import MergePipeline
from orbitalengineer.engine.orbitalcl.interaction.interaction import InteractionGroupPipeline
from orbitalengineer.engine.orbitalcl.nudge.nudge import NudgePipeline
from orbitalengineer.engine.orbitalcl.position.position import PositionPipeline
from orbitalengineer.engine.orbitalcl.velocity.velocity import VelocityPipeline
from orbitalengineer.engine.orbitalcl.bounce.bounce import BouncePipeline

from orbitalengineer.helpers import r_from_mass
from orbitalengineer.ipc import message

mf = cl.mem_flags
kernel_dir = Path(__file__).parent

class SimController_CL:
    coef_of_restitution = config.COEF_OF_RESTITUTION
    dt_base = config.DEFAULT_DT_BASE
    G = config.GRAV_CONSTANT
    EPS_DIST:float = config.EPS_DIST
    EPS_TIME:float = config.EPS_TIME
    N:int = 0
    
    shm:dict[str, shared_memory.SharedMemory] = dict()
    
    def __init__(self):
        self.enable_profiling = config.EMIT_METRICS
        self.tr = EventTracer(self)
        self.metrics = MetricsProducer(config.METRIC_SOCKET_PATH)
        self.device = None
        self.reset()

    def reset(self):
        self.accum = 0.0
        self.last_now:float|None = None
        self.is_initialized = False
        self.tick_id = 0
        self.step_count = 0

    def _shared_memory(self, field_name:str, size:int, dtype:type) -> NDArray:
        t = np.dtype(dtype)
        logger.info("shm: %s size=%s dtype=%s", field_name, size, t)
        self.shm[field_name] = shared_memory.SharedMemory(create=True, size=t.itemsize * size)
        array = np.ndarray(size, dtype=dtype, buffer=self.shm[field_name].buf)
        return array

    def disconnect(self):
        if not hasattr(self, 'shm'):
            return
        closed = []
        for name, shm in self.shm.items():
            try:
                shm.close()
                shm.unlink()
                closed.append(name)
            except FileNotFoundError:
                logger.warning("Could not properly close shared memory: file(s) not found.")
                ...
        logger.info("Closed shared memory: %s", ','.join(closed))

    @log_timing
    def _populate_particle_fields(self, particles:Sequence[message.ParticleInit]):
        for i,p in enumerate(particles):
            self.flags[i] = np.uint32(p.flags)
            self.velocity[i] = np.complex64(*p.velocity)
            self.position[i] = np.complex64(*p.position)
            self.mass[i] = np.float32(p.mass)
            self.radius[i] = np.float32(p.radius)
        logger.info("Populated %s bodies", len(particles))

    @log_timing
    def _allocate_memory(self):
        self.flags = self._shared_memory('flags', self.N, np.uint32)
        self.velocity = self._shared_memory('velocity', self.N, np.complex64)
        self.position = self._shared_memory('position', self.N, np.complex64)
        self.mass = self._shared_memory('mass', self.N, np.float32)
        self.radius = self._shared_memory('radius', self.N, np.float32)
        self.force = self._shared_memory('force', self.N * self.N, np.complex64)
        self.cgroup = self._shared_memory('cgroup', self.N, np.uint32)
        self.cgroup[:] = np.arange(self.N, dtype=np.uint32)

    def _create_buffers(self):
        self.flags_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.flags)
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.pos_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.position)
        self.mass_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.mass)
        self.radius_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.radius)
        self.force_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.force)
        self.cgroup_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.cgroup)

    @log_timing
    def _init_queue(self):
        properties = cast(cl.command_queue_properties, 0)
        if self.enable_profiling:
            logger.info("OpenCL profiling enabled.")
            properties = cl.command_queue_properties.PROFILING_ENABLE
        self.q = cl.CommandQueue(self.ctx, properties=properties)
    
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
        logger.info("Using build constants: %s", [ f'{k}={v}' for k,v in defs.items() ])
        return [ f'-D{k}={v}' for k,v in defs.items() ]
    
    @log_timing
    def _init_kernel(self):
        if self.device is None:
            #from warnings import warn
            #warn("Cannot initialize kernels until CL device is selected.", stacklevel=2)
            raise InitKernelException()
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
            self._cgroup = CGroupPipeline(self.N, self.ctx, self.q, self.tr, build_options)
        except (cl._cl.RuntimeError, cl._cl.LogicError) as e: #type:ignore
            logger.error(str(e))
            return False
        return True
    
    def set_cl_device(self, platform_id:int, device_id:int):
        if self.is_initialized:
            logger.warning("set_cl_device called after initialization.")
        self.device = CLDeviceManager(platform_id, device_id)
        self.ctx = cl.Context([self.device._device])        
        logger.info("CL device set to (%s, %s)", platform_id, device_id)
    
    @log_timing
    def init_sim(self, particles:Sequence[message.ParticleInit]):
        self.N = len(particles)
        self._allocate_memory()
        self._populate_particle_fields(particles)
        self._init_queue()
        if not self._init_kernel():
            return False
        self._create_buffers()
        self._interaction(self.dt_base, self.flags_cl, self.pos_cl, self.vel_cl, self.radius_cl, self.mass_cl)
        if config.NUDGE_ON_START_ENABLE:
            for i in range(10):
                self.nudge()
        self.is_initialized = True
        return True
    
    def apply_vector_offset(self, vector_name:str, ids:Sequence[int], op:str, offset:tuple[float,float]):
        if vector_name == "position":
            vector = self.position
            buffer = self.pos_cl
            value = np.complex64(offset[0], offset[1])
        elif vector_name == "velocity":
            vector = self.velocity
            buffer = self.vel_cl
            value = np.complex64(offset[0], offset[1])
        elif vector_name == "mass":
            vector = self.mass
            buffer = self.mass_cl
            value = np.float32(offset[0])
        elif vector_name == "radius":
            vector = self.radius
            buffer = self.radius_cl
            value = np.float32(offset[0])
        else:
            logger.error("Invalid vector name for apply_vector_offset. "
                         "Must be one of: position, velocity, mass, or radius.")
            return False
        
        if op == 'add':
            vector[ids] += value
        elif op == 'mul':
            vector[ids] *= value
        cl.enqueue_copy(self.q, buffer, vector)
        
        if vector_name == "mass":
            self.radius[ids] = np.vectorize(r_from_mass)(self.mass[ids])
            cl.enqueue_copy(self.q, self.radius_cl, self.radius) 
        
        return True
    
    def _has_queue(self) -> bool:
        return hasattr(self, 'q')

    def sync(self):
        if not self._has_queue():
            logger.warning("Cannot sync() prior to a queue being configured.")
            return
        self.q.finish()
        cl.enqueue_copy(self.q, self.flags, self.flags_cl)
        cl.enqueue_copy(self.q, self.position, self.pos_cl)
        cl.enqueue_copy(self.q, self.velocity, self.vel_cl)
        cl.enqueue_copy(self.q, self.mass, self.mass_cl)
        cl.enqueue_copy(self.q, self.radius, self.radius_cl)
        #cl.enqueue_copy(self.q, self._interaction.toi, self._interaction.toi_cl)
        cl.enqueue_copy(self.q, self.cgroup, self.cgroup_cl)
        self.q.finish()
    
    def nudge(self):
        return self._nudge(self.flags_cl, self.pos_cl, self.mass_cl, self.radius_cl)
    
    def kick(self, dt_step):
        return self._velocity(dt_step, self.flags_cl, self.pos_cl, self.mass_cl, self.radius_cl, self.vel_cl, self.force_cl)
    
    def drift(self, dt_step):
        return self._position(dt_step, self.flags_cl, self.vel_cl, self.pos_cl)
    
    def _minimum_viable_dt(self, dt_step):
        cl.enqueue_copy(self.q, self._interaction.node_dt, self._interaction.node_dt_cl).wait()
        try:
            min_toi = min([t for t in self._interaction.node_dt if t > 0])
        except ValueError:
            min_toi = dt_step
        return max(min(dt_step, min_toi), config.EPS_TIME)
    
    def single_step(self, dt_step):
        initial_dt_step = dt_step
        count = 0
        while dt_step > config.EPS_TIME and count < 10:
            dt = self._minimum_viable_dt(dt_step)
            
            self.kick(dt / 2.0)
            self.drift(dt)
            self.kick(dt / 2.0)
            
            self._cgroup(self.flags_cl, self.pos_cl, self.radius_cl, self._interaction.toi_cl, self.cgroup_cl)
                        
            if config.COLLISION_MERGE_ENABLE:
                self._merge(self.flags_cl, self.cgroup_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl)
            
            if config.COLLISION_BOUNCE_ENABLE:
                self._bounce(self.flags_cl, self.pos_cl, self.vel_cl, self.mass_cl, self.radius_cl)

            self._interaction(dt, self.flags_cl, self.pos_cl, self.vel_cl, self.radius_cl, self.mass_cl)

            dt_step -= dt
            self.step_count += 1
            count += 1
        
        if count >= config.MAX_SUB_STEPS and dt_step > config.EPS_TIME:
            logger.warning(f"Over-iterated step. {dt_step=}, {initial_dt_step=}")
        
        return count, dt_step
    
    def emit_metrics(self, dt_step_size:float):
        if not config.EMIT_METRICS or not self.enable_profiling:
            return
        self.q.finish()
        timeline = self.tr.timeline()
        self.tr.clear()
        self.metrics.emit_metric("tick", self.tick_id, timeline=[
            { "name":t["name"], "duration_ms":round(t["duration_ns"]/1e6, 6)}
            for t in timeline
        ], dt_step=round(float(dt_step_size), 6))

    def tick(self, now:float):
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
            try:
                count, dt_unprocessed = self.single_step(dt_step)
            except (cl._cl.RuntimeError, cl._cl.LogicError) as e: #type:ignore
                import sys
                print(e, file=sys.stderr)
                raise e
                return False
            self.accum -= dt_step
            #self.accum -= dt_unprocessed # type: ignore
            #self.accum -= (dt_step - dt_unprocessed) # type: ignore
            self.emit_metrics(float(dt_step))
            self.tick_id += 1
        return num_steps

    # def to_dict(self) -> dict:
    #     self.sync()
    #     return {
    #         # Simulation state
    #         "tick_id": int(self.tick_id),
    #         "dt_base": float(self.dt_base),
    #         "step_count": int(self.step_count),
    #         "Lx": int(self.Lx),
    #         "N": int(self.N),
            
    #         # Constants
    #         "G": float(self.G),
    #         "coef_of_restitution": float(self.coef_of_restitution),
    #         "EPS_DIST": float(self.EPS_DIST),
    #         "EPS_TIME": float(self.EPS_TIME),
            
    #         # particle field vectors
    #         "flags":    [int(fl) for fl in self.flags],
    #         "position": [(f"{p.real:.6f}", f"{p.imag:.6f}") for p in self.position],
    #         "velocity": [(f"{v.real:.6f}", f"{v .imag:.6f}") for v in self.velocity],
    #         "mass":     [f"{m:.6f}" for m in self.mass],
    #         "radius":   [f"{r:.6f}" for r in self.radius],
    #     }
    
    def load_from_dict(self, obj:dict):
        for field in ['tick_id', 'dt_base', 'step_count', 'Lx', 'N', 'G', 'EPS_DIST', 'EPS_TIME']:
            setattr(self, field, obj[field])
        
        def vector_complex64(values:list):
            return np.array([np.complex64(float(v[0]), float(v[1])) for v in values], dtype=np.complex64)

        def vector_float32(values:list):
            return np.array([np.float32(v) for v in values], dtype=np.float32)

        def vector_uint32(values:list):
            return np.array([np.uint32(v) for v in values], dtype=np.uint32)
        
        self._allocate_memory()
        self.flags[:] = vector_uint32(obj['flags'])
        self.position[:] = vector_complex64(obj['position'])
        self.velocity[:] = vector_complex64(obj['velocity'])
        self.mass[:] = vector_float32(obj['mass'])
        self.radius[:] = vector_float32(obj['radius'])
        return self