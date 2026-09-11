import os
from pathlib import Path
from typing import Any, Sequence, TypeAlias, cast
import numpy as np
from numpy.typing import NDArray
import pyopencl as cl
from pyopencl import typing

from orbitalengineer.engine.orbitalcl.named_shared_memory import NamedSharedMemory
from orbitalengineer.engine.orbitalcl.sim_state import SimState
from orbitalengineer.engine.orbitalcl.tracer import EventTracer

_ExtendedKernelArg: TypeAlias = """typing.KernelArg | cl.MemoryObjectHolder"""

class OrbitalKernel(cl.Kernel):
    def __init__(self, arg0: cl.Program, arg1: str, tr: EventTracer, /):
        super().__init__(arg0, arg1)
        self.tr = tr
    
    def __call__(
        self,
        queue: cl.CommandQueue,
        global_work_size: tuple[int, ...],
        local_work_size: tuple[int, ...] | None,
        *args: _ExtendedKernelArg,
        wait_for: typing.WaitList = None,
        g_times_l: bool = False,
        allow_empty_ndrange: bool = False,
        global_offset: tuple[int, ...] | None = None,
    ) -> cl.Event:
        #print(self.function_name, flush=True)
        return self.tr.add(
            self.function_name,
            super().__call__(
                queue,
                global_work_size,
                local_work_size,
                *args, # type: ignore
                wait_for=wait_for,
                g_times_l=g_times_l,
                allow_empty_ndrange=allow_empty_ndrange,
                global_offset=global_offset
            )
        ).wait()

class PipelineComponent:
    Lx:int = 256
    debug_flag:str = '_'
    
    
    pipeline_run_id:int = 0
    dependencies:list['PipelineComponent']|None = None 
    
    def __init__(self, shm:NamedSharedMemory, pipeline_state:SimState, ctx:cl.Context, queue:cl.CommandQueue, tr:EventTracer, build_options:Sequence|None=None):
        self.N = pipeline_state.N
        if self.N < self.Lx: self.Lx = self.N
        
        self.shm = shm
        self.default_build_options = build_options or ['-cl-std=CL2.0']
        self.ctx = ctx
        self.queue = queue
        self.tr = tr
        self._host_vector:dict[cl.Buffer, NDArray] = dict()
        self._check_debug_flag()
        self.initialize()
        
    def alloc(self, size:int, dtype:type, fill:Sequence|None=None, shared_name:None|str=None) -> cl.Buffer:
        if shared_name is not None:
            vec = self.shm.create_shared_memory(shared_name, size, dtype)
        elif fill is not None:
            vec = np.array(fill, dtype=dtype)
        else:
            vec = np.zeros(size, dtype)
        b = self._create_buffer(vec)
        self._host_vector[b] = vec
        return b
    
    def get_host_vector(self, buffer: cl.Buffer, sync:bool=False) -> NDArray:
        vec = self._host_vector[buffer]
        if sync:
            self.sync_to_host(buffer).wait()
        return vec
    
    def sync_to_host(self, buffer: cl.Buffer) -> cl.Event:
        return cl.enqueue_copy(self.queue, self.get_host_vector(buffer), buffer)
    
    def sync_to_device(self, buffer: cl.Buffer) -> cl.Event:
        return cl.enqueue_copy(self.queue, buffer, self.get_host_vector(buffer))
    
    # def add_dependency(self, dep: 'CLPipelineStep'):
    #     if self.dependencies is None:
    #         self.dependencies = []
    #     if dep not in self.dependencies:
    #         self.dependencies.append(dep)
    
    # def _run_dependencies(self):
    #     if self.dependencies is None:
    #         return
    #     for dep in self.dependencies:
    #         if dep.pipeline_run_id < self.pipeline_run_id:
    #             dep.run_pipeline_step(self.pipeline_run_id)
    
    def _check_debug_flag(self):
        debug_flags = os.environ.get("DEBUG", "").lower()
        if debug_flags == "all" or self.debug_flag.lower() in debug_flags:
            self.default_build_options = [
                *self.default_build_options,
                f"-DDEBUG=true"
            ]
    
    def _create_buffer(self, hostbuf) -> cl.Buffer:
        mf = cl.mem_flags
        return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf)
    
    def _get_kernel_src(self, kernel_file: str):
        with open(Path(__file__).parent / kernel_file, 'r') as f:
            return f.read()
    
    def _load_kernel(self, kernel_name:str, kernel_file:str, build_options:Sequence|None=None):
        opts = build_options or self.default_build_options
        kernel_src = self._get_kernel_src(kernel_file)
        prg = cl.Program(self.ctx, kernel_src).build(options=[*opts])
        return OrbitalKernel(prg, kernel_name, self.tr)

    def initialize(self):
        raise NotImplementedError()