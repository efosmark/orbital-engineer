import time
from typing import Sequence, cast
from pathlib import Path
import pyopencl as cl

from orbitalengineer.engine import log_timing, logger
from orbitalengineer.engine.exception import InitKernelException
from orbitalengineer.engine.metric import MetricsProducer

from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.orbitalcl.ledger.ledger import LedgerController
from orbitalengineer.engine.orbitalcl.named_shared_memory import NamedSharedMemory
from orbitalengineer.engine.orbitalcl.cgroup.cgroup import CGroupPipeline
from orbitalengineer.engine.orbitalcl.device import CLDeviceManager
from orbitalengineer.engine.orbitalcl.distance.distance import DistancePipeline
from orbitalengineer.engine.orbitalcl.sim_state import SimState
from orbitalengineer.engine.orbitalcl.primary_vectors import PrimaryStateVectors
from orbitalengineer.engine.orbitalcl.sim_config import SimConfig
from orbitalengineer.engine.orbitalcl.tracer import EventTracer
from orbitalengineer.engine.orbitalcl.merge.merge import MergePipeline
from orbitalengineer.engine.orbitalcl.interaction.interaction import InteractionPipeline
from orbitalengineer.engine.orbitalcl.nudge.nudge import NudgePipeline
from orbitalengineer.engine.orbitalcl.position.position import PositionPipeline
from orbitalengineer.engine.orbitalcl.velocity.velocity import VelocityPipeline
from orbitalengineer.engine.orbitalcl.bounce.bounce import BouncePipeline
from orbitalengineer.engine.orbitalcl.velocity_along_normal.velocity_along_normal import VelocityAlongNormalPipeline
from orbitalengineer.ipc import message

mf = cl.mem_flags
kernel_dir = Path(__file__).parent

class SimController_CL:
    cfg:SimConfig
    last_tick_at:float|None = None
    
    def __init__(self):
        self.tr = EventTracer(self)
        self.reset()
        self.metrics = MetricsProducer(self.cfg.METRIC_SOCKET_PATH)

    def reset(self):
        self.is_initialized = False
        self.cfg = SimConfig()
        self.pipeline_state = SimState()
        self.shm = NamedSharedMemory()

    def _init_queue(self):
        properties = cast(cl.command_queue_properties, 0)
        if self.cfg.ENABLE_PROFILING:
            logger.info("OpenCL profiling enabled.")
            properties = cl.command_queue_properties.PROFILING_ENABLE
        self.q = cl.CommandQueue(self.ctx, properties=properties)
        self.copy_queue = cl.CommandQueue(self.ctx, properties=properties)
    
    def _generate_headers(self):
        build_dir = Path(".opencl_build")
        build_dir.mkdir(exist_ok=True)
        header_path = build_dir / "flags.clh"
        header_path.write_text(flags.generate_cl_flag_file())
        return build_dir
    
    @log_timing
    def _init_kernels(self):
        if self.device is None:
            #from warnings import warn
            #warn("Cannot initialize kernels until CL device is selected.", stacklevel=2)
            raise InitKernelException()
        build_dir = self._generate_headers()                
        
        build_options = [
            '-cl-std=CL2.0',
            f'-I {kernel_dir}',
            f'-I {build_dir}',
            *self.cfg.as_build_contants()
        ]
        
        if self.cfg.USE_FAST_RELAXED_MATH:
            build_options.append('-cl-fast-relaxed-math')
        
        args = [self.shm, self.vec, self.ctx, self.q, self.copy_queue, self.tr, build_options]        
        try:
            self._ledger = LedgerController(*args)
            self._velocity = VelocityPipeline(*args)
            self._position = PositionPipeline(*args)
            self._bounce = BouncePipeline(*args)
            self._interaction = InteractionPipeline(*args)
            self._nudge = NudgePipeline(*args)
            self._merge = MergePipeline(*args)
            self._cgroup = CGroupPipeline(*args)
            self._distance = DistancePipeline(*args)
            self._velocity_along_normal = VelocityAlongNormalPipeline(*args)
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
        self.pipeline_state.N = len(particles)
        self.pipeline_state.tick_id = 0
        self.pipeline_state.step_id = 0
        self._init_queue()
        self.vec = PrimaryStateVectors(self.shm, self.ctx, self.q, self.tr)
        self.vec.initialize(len(particles))
        self.vec.populate(particles)
        
        if not self._init_kernels():
            return False
        
        self._interaction(self.cfg.DEFAULT_DT_BASE, self.vec)
        if self.cfg.NUDGE_ON_START_ENABLE:
            for i in range(10):
                self.nudge()
        self.is_initialized = True
        return True

    def _has_queue(self) -> bool:
        return hasattr(self, 'q')
    
    def nudge(self):
        return self._nudge(self.vec)
    
    def kick(self, dt_step):
        with self.tr('wait-for-kick (host)'):
            self._velocity(dt_step, self.vec, self._distance)
            self._velocity.queue.finish()
    
    def drift(self, dt_step):
        with self.tr('wait-for-drift (host)'):
            self._position(dt_step, self.vec)
            self._position.queue.finish()
    
    def compute_interaction_dt(self, min_dt:float):
        with self.tr('wait-for-dt (host)'):
            self._interaction(self.cfg.DEFAULT_DT_BASE, self.vec)
            self._interaction.queue.finish()
    
    def substep(self, dt_step):
        dt = self._interaction.minimum_viable_dt(dt_step, self.cfg.EPS_TIME)
        self.pipeline_state.step_id += 1

        self.kick(dt / 2.0)
        self.drift(dt)

        self._distance(self.vec)
        
        if self.cfg.COLLISION_MERGE_ENABLE:
           self._merge(self.vec, self._distance, self._ledger)
        
        if self.cfg.COLLISION_BOUNCE_ENABLE:
            self._velocity_along_normal(self.vec)
            self._bounce(self.vec, self._interaction, self._cgroup, self._distance, self._velocity_along_normal, self._ledger)

        with self.tr('wait-for-collisions (host)'):
            self.q.finish()

        with self.tr('wait-for-cgroup (host)'):
            self._cgroup(self._distance)
            self.q.finish()

        self.kick(dt / 2.0)
        self.compute_interaction_dt(dt)
        self.vec.defragment()
        self._ledger.commit(self.pipeline_state.tick_id, self.pipeline_state.step_id) #.wait()
        return dt
    
    def single_step(self, dt_step):
        logger.debug("single_step(dt_step=%.4f)", dt_step)
        count = 0
        while dt_step > self.cfg.EPS_TIME and count < self.cfg.MAX_SUB_STEPS:
            dt = self.substep(dt_step)
            dt_step -= dt
            self.pipeline_state.step_id += 1
            count += 1
        return count, dt_step

    def tick(self, dt_step:float):
        if self.last_tick_at is None:
            self.last_tick_at = time.monotonic_ns()
        if not self.is_initialized:
           logger.warning("tick() was called before simulation initialization.")
           return 0, 0
        logger.debug("tick(dt_step=%.4f)", dt_step)

        try:
            count, dt_unprocessed = self.single_step(dt_step)
            logger.debug("%.0f, %.3f = single_step(%.4f)", count, dt_unprocessed, dt_step)
        except (cl._cl.RuntimeError, cl._cl.LogicError) as e: #type:ignore
            import sys
            print(e, file=sys.stderr)
            raise e

        if count > 0:
            self.emit_metrics(float(dt_step))
            self.pipeline_state.tick_id += 1
            self.pipeline_state.step_id = 0
        return count, dt_unprocessed
    
    def emit_metrics(self, dt_step_size:float):
        with self.tr('emit_metrics (host)'):
            if not self.cfg.ENABLE_PROFILING: return
            self.q.finish()
    
            timeline = self.tr.timeline()
            self.tr.clear()
        
            self.metrics.emit_metric("tick", self.pipeline_state.tick_id, timeline=[
                { "name":t.name, "duration_ms":round(t.duration_ns/1e6, 6)}
                for t in timeline
            ], dt_step=round(float(dt_step_size), 6))

    def sync(self):
        self.vec.sync(self.copy_queue)
        self.vec.sync_to_host(self._ledger.ledger, queue=self.copy_queue)
        self.vec.sync_to_host(self._cgroup.group, queue=self.copy_queue)
        self.vec.sync_to_host(self._distance.direct_contacts, queue=self.copy_queue)
        self.vec.sync_to_host(self._distance.n_direct_contacts, queue=self.copy_queue)            
        self.copy_queue.finish()
        self.q.finish()
