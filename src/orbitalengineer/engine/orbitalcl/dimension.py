import os
from pathlib import Path
from typing import Sequence
import pyopencl as cl
from numpy.typing import NDArray

from orbitalengineer.engine.orbitalcl.tracer import EventTracer

mf = cl.mem_flags


def get_kernel_src(kernel_file: str):
    with open(Path(__file__).parent / kernel_file, 'r') as f:
        return f.read()

def load_kernel(kernel_name:str, kernel_file:str, ctx:cl.Context, build_options:Sequence|None=None):
    opts = build_options or ['-cl-std=CL2.0']
    kernel_src = get_kernel_src(kernel_file)
    prg = cl.Program(ctx, kernel_src).build(options=[*opts])
    return cl.Kernel(prg, kernel_name)


class CLPipelineStep:
    Lx:int = 256
    debug_flag:str = '_'
    
    def __init__(self, N:int, ctx:cl.Context, queue:cl.CommandQueue, tr:EventTracer, build_options:Sequence|None=None):
        self.N = N
        
        self.default_build_options = build_options or ['-cl-std=CL2.0']
        self.ctx = ctx
        self.queue = queue
        self.tr = tr
        self._check_debug_flag()
        self.initialize()
    
    def _check_debug_flag(self):
        debug_flags = os.environ.get("DEBUG", "").lower()
        if debug_flags == "all" or self.debug_flag.lower() in debug_flags:
            self.default_build_options = [
                *self.default_build_options,
                f"-DDEBUG=true"
            ]
    
    def _create_buffer(self, hostbuf) -> cl.Buffer:
        return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf)
    
    def _load_kernel(self, kernel_name:str, kernel_file:str, build_options:Sequence|None=None):
        opts = build_options or self.default_build_options
        kernel_src = get_kernel_src(kernel_file)
        prg = cl.Program(self.ctx, kernel_src).build(options=[*opts])
        return cl.Kernel(prg, kernel_name)

    def initialize(self):
        raise NotImplementedError()