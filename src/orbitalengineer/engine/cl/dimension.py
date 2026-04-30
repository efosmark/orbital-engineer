from pathlib import Path
from typing import Sequence
import pyopencl as cl
from numpy.typing import NDArray

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
    
    def __init__(self, N:int, ctx:cl.Context, queue:cl.CommandQueue, build_options:Sequence|None=None):
        self.N = N
        
        self.default_build_options = build_options or ['-cl-std=CL2.0']
        self.ctx = ctx
        self.queue = queue
        self.init_memory(N)
        self.load_kernels()
    
    def _create_buffer(self, hostbuf) -> cl.Buffer:
        return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf)
    
    def _load_kernel(self, kernel_name:str, kernel_file:str, build_options:Sequence|None=None):
        opts = build_options or self.default_build_options
        kernel_src = get_kernel_src(kernel_file)
        prg = cl.Program(self.ctx, kernel_src).build(options=[*opts])

        return cl.Kernel(prg, kernel_name)

    def load_kernels(self):...
    
    def init_memory(self, N:int): ...
    
    def sync_host_to_device(self):...
    
    def sync_device_to_host(self):...
    
    def compute(self, *args, **kwargs) -> cl.Event: ...
    
    