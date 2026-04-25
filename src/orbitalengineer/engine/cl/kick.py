import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.kernelizer import KernelBuilder
mf = cl.mem_flags

class KickPairwise(KernelBuilder):
    kernel_name = "kick_pairwise"
    kernel_file = "kernel/kick.clh"

    def __call__(
        self,
        N: int, 
        dt_value: float,
        pos: cl.Buffer,
        vel: cl.Buffer,
        mass: cl.Buffer,
        radius: cl.Buffer,
        toi: cl.Buffer,
        min_impact_time: cl.Buffer,
        interactions: cl.Buffer,
        *,
        local_size:int=64,
        metric_alias:str|None=None
    ):
        
        Lx = min(N, int(local_size))
        return self.kernel(
            self.queue,
            
            (N * Lx, ), # global work size
            (Lx, ),     # local work size
            
            # Args
            np.uint32(N),
            np.float32(dt_value),
            pos, vel, mass, radius,
            toi, min_impact_time, interactions,
            
            metric_alias=metric_alias
        )