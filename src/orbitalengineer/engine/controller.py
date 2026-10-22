import time
from typing import Sequence, cast
from pathlib import Path
import psutil
import pyopencl as cl
import numpy as np

from orbitalengineer import dtypes, flags, body
from orbitalengineer.engine import log_timing, logger
from orbitalengineer.engine.energy.energy import EnergyPipeline
from orbitalengineer.engine.exception import InitKernelException
from orbitalengineer.engine.metric import MetricsProducer

from orbitalengineer.engine.ledger.ledger import LedgerController
from orbitalengineer.engine.named_shared_memory import NamedSharedMemory
from orbitalengineer.engine.cgroup.cgroup import CGroupPipeline
from orbitalengineer.engine.device import CLDeviceManager
from orbitalengineer.engine.distance.distance import DistancePipeline
from orbitalengineer.engine.sim_state import SimState
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.engine.tracer import EventTracer
from orbitalengineer.engine.merge.merge import MergePipeline
from orbitalengineer.engine.interaction.interaction import InteractionPipeline
from orbitalengineer.engine.nudge.nudge import NudgePipeline
from orbitalengineer.engine.position.position import PositionPipeline
from orbitalengineer.engine.velocity.velocity import VelocityPipeline
from orbitalengineer.engine.bounce.bounce import BouncePipeline
from orbitalengineer.engine.velocity_along_normal.velocity_along_normal import VelocityAlongNormalPipeline
from orbitalengineer.ipc import message

mf = cl.mem_flags
kernel_dir = Path(__file__).parent

class SimController:
    cfg:SimConfig
    last_tick_at:float|None = None
    
    def __init__(self):
        self.tr = EventTracer(self)
        self.reset()
        self.metrics = MetricsProducer(self.cfg.METRIC_SOCKET_PATH)
        self.cpu_num = psutil.Process().cpu_num()
        self._cpu_last_sample = 0
        self._cpu_utilization = 0

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
    
    def _generate_headers(self):
        build_dir = Path(".opencl_build")
        build_dir.mkdir(exist_ok=True)
        header_path = build_dir / "flags.clh"
        header_path.write_text(flags.generate_cl_flag_file())
        
        header_path = build_dir / "dtype.clh"
        header_path.write_text(dtypes.generate_cl_dtype_file())
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
        
        args = [self.shm, self.vec, self.ctx, self.q, self.q, self.tr, build_options]        
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
            self._energy = EnergyPipeline(*args)
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
    def init_sim(self, particles:Sequence[body.Body]):
        self.pipeline_state.N = len(particles)
        self.pipeline_state.tick_id = 0
        self.pipeline_state.step_id = 0
        self._init_queue()
        self.vec = PrimaryStateVectors(self.shm, self.ctx, self.q, self.tr)
        self.vec.initialize(len(particles))
        self.vec.populate(particles)
        
        if not self._init_kernels(): return False
        
        self._interaction(self.cfg.DEFAULT_DT_BASE, self.vec)
        if self.cfg.NUDGE_ON_START_ENABLE:
            for i in range(10):
                self.nudge()
        
        self._energy(self.vec)
        
        total_KE, total_GPE = self.get_energy()
        self._starting_total_energy = total_KE + total_GPE
        
        self.is_initialized = True
        return True

    def _has_queue(self) -> bool:
        return hasattr(self, 'q')
    
    def nudge(self):
        """
        Attempt to push away any overlapping bodies.
        This is useful during startup as a means to ensure randomly-placed bodies don't occupy the same space.
        """
        return self._nudge(self.vec)
    
    def kick(self, dt_step):
        """
        Run the kick step of Leapfrog KDK.
        Updates the velocity for each body, based on the interactions between every other body.
        
        This expects the value of `dt_step` to already be 1/2 the full step value.
        """
        with self.tr('wait-for-kick (host)'):
            self._velocity(dt_step, self.vec, self._distance)
            self._velocity.queue.finish()
    
    def drift(self, dt_step):
        """Run the drift step of Leapfrog KDK in order to update the body positions."""
        with self.tr('wait-for-drift (host)'):
            self._position(dt_step, self.vec)
            self._position.queue.finish()
    
    def compute_interaction_dt(self):
        """Determine the interaction time for each body.
        
        Interaction time is defined as when the body will collide with another body.
        """
        with self.tr('wait-for-dt (host)'):
            self._interaction(self.cfg.DEFAULT_DT_BASE, self.vec)
            self._interaction.queue.finish()
    
    def substep(self, dt_step):
        if self.cfg.COLLISION_MERGE_ENABLE or self.cfg.COLLISION_BOUNCE_ENABLE:
            dt = self._interaction.minimum_viable_dt(dt_step, self.cfg.EPS_TIME)
        else:
            dt = dt_step
        
        self.kick(dt / 2.0)
        self.drift(dt)
        self.kick(dt / 2.0)

        self._distance(self.vec)
        
        if self.cfg.COLLISION_MERGE_ENABLE:
           self._merge(self.vec, self._distance, self._ledger)
        
        if self.cfg.COLLISION_BOUNCE_ENABLE:
            self._velocity_along_normal(self.vec)
            self._bounce(self.vec, self._interaction, self._cgroup, self._distance, self._velocity_along_normal, self._ledger)

        with self.tr('wait-for-collisions (host)'):
            self.q.finish()

        # with self.tr('wait-for-cgroup (host)'):
        #     self._cgroup(self._distance)
        #     self.q.finish()

        if self.cfg.COLLISION_MERGE_ENABLE or self.cfg.COLLISION_BOUNCE_ENABLE:
            self.compute_interaction_dt()
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
            self._energy(self.vec)
            self.pipeline_state.tick_id += 1
            self.pipeline_state.step_id = 0        
        return count, dt_unprocessed        
    
    def get_energy(self):
        ke_buffer = self._energy.kinetic_energy
        kinetic_energy = self.vec.get_host_vector(self._energy.kinetic_energy)
        
        gpe_buffer = self._energy.grav_potential_energy
        grav_potential_energy = self.vec.get_host_vector(self._energy.grav_potential_energy)

        cl.enqueue_copy(self.q, kinetic_energy, ke_buffer).wait()
        cl.enqueue_copy(self.q, grav_potential_energy, gpe_buffer).wait()
        
        total_KE = float(np.sum(kinetic_energy, dtype=np.float64))
        total_GPE = float(np.sum(grav_potential_energy, dtype=np.float64))
        return total_KE, total_GPE
    
    def get_total_energy_error(self):
        total_KE, total_GPE = self.get_energy()
        total_E = total_KE + total_GPE
        return (total_E - self._starting_total_energy) / self._starting_total_energy
        
    def get_cpu_utilization(self):
        now = time.monotonic()
        if now - self._cpu_last_sample >= 0.3:
            self._cpu_utilization = psutil.cpu_percent(
                interval=None,
                percpu=True
            )[self.cpu_num]
            self._cpu_last_sample = now
        return self._cpu_utilization

    def sync(self, current_buffer):
        self.vec.sync(self.q, current_buffer)
        self.vec.sync_to_host(self._ledger.ledger, queue=self.q)
        #self.vec.sync_to_host(self._cgroup.group, queue=self.q)
        self.vec.sync_to_host(self._distance.direct_contacts, queue=self.q)
        self.vec.sync_to_host(self._energy.kinetic_energy, queue=self.q)
        self.vec.sync_to_host(self._energy.grav_potential_energy, queue=self.q)
        self.q.finish()
