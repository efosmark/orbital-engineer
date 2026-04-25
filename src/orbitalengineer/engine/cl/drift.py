import numpy as np
import pyopencl as cl

from orbitalengineer.engine.cl.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src_pairwise = """
#include "kernel/complex.clh"

__kernel void drift(
               const float   dt,
    __global   cfloat* velocity,
    __global   cfloat* position,
    __global   float*  time_to_impact
) {
    uint i = get_global_id(0);
    position[i] += velocity[i] * dt;
}
"""

class Drift(KernelBuilder):
    kernel_name = "drift"
    kernel_src = kernel_src_pairwise

    def __call__(
        self,
        N: int, 
        dt_step: float,
        velocity: cl.Buffer,
        position: cl.Buffer,
        time_to_impact: cl.Buffer
    ):
        return self.kernel(
            self.queue,
            (N,),
            None,
                        
            # Args
            np.float32(dt_step),
            velocity,
            position,
            time_to_impact
        )
