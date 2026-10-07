import os
from pathlib import Path
from typing import Any, Sequence, TypeAlias, cast
import numpy as np
from numpy.typing import NDArray
import pyopencl as cl
from pyopencl import typing


from orbitalengineer.engine.named_shared_memory import NamedSharedMemory
from orbitalengineer.engine.primary_vectors import PrimaryStateVectors
from orbitalengineer.engine.tracer import EventTracer

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
        ) #.wait()

class PipelineComponent:
    debug_flag:str = '_'    
    dependencies:list['PipelineComponent']|None = None 
    
    def __init__(self, shm:NamedSharedMemory, vec:PrimaryStateVectors, ctx:cl.Context, queue:cl.CommandQueue, copy_queue:cl.CommandQueue, tr:EventTracer, build_options:Sequence|None=None):
        self.shm = shm
        self.vec = vec
        self.ctx = ctx
        self.queue = queue
        self.copy_queue = copy_queue
        self.default_build_options = build_options or ['-cl-std=CL2.0']
        self.tr = tr
        self._host_vector:dict[cl.Buffer, NDArray] = dict()
        self._check_debug_flag()
        self.initialize()
    
    def _check_debug_flag(self):
        debug_flags = os.environ.get("DEBUG", "").lower()
        if debug_flags == "all" or self.debug_flag.lower() in debug_flags:
            self.default_build_options = [
                *self.default_build_options,
                f"-DDEBUG=true"
            ]
    
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