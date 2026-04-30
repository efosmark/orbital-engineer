import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
__kernel void global_row_reduce(
             const uint increment,
    __global const float2* partials,
    __global float2 *restrict result
) {
    uint N  = get_global_size(0);
    uint Lx = get_local_size(1);

    uint row = get_global_id(0);
    uint g1 = get_global_id(1);

    // Work-item grid stride summing
    float2 accum = (float2) (0, 0);
    uint base = row * N;
    for (uint idx = base + g1; idx < base + N; idx += Lx) {
        accum += partials[idx];
    }

    // Sub-group reduction
    float real = sub_group_reduce_add(accum.x);
    float imag = sub_group_reduce_add(accum.y);
    if (get_sub_group_local_id() != 0) {
        real = 0.0f;
        imag = 0.0f;
    }

    // Work-group reduction
    float sum_real = work_group_reduce_add(real);
    float sum_imag = work_group_reduce_add(imag);

    // Work-group leader writes result
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        if (increment)
            result[row] += (float2)(sum_real, sum_imag);
        else
            result[row] = (float2)(sum_real, sum_imag);
    }
}
"""

class Float2RowReducer(KernelBuilder):
    kernel_name = "global_row_reduce"
    kernel_src = kernel_src
    
    def __call__(self, N:int, input_table:cl.Buffer, output_buffer:cl.Buffer, *, local_size:int=256, increment=True, metric_alias:str|None=None):
        Lx = int(local_size)
        return self.kernel(
            self.queue,
            (N, Lx), # global work size
            (1, Lx), # local work size
            
            # Args
            (np.uint32(1) if increment else np.uint32(0)),
            input_table,
            output_buffer,
            
            metric_alias=metric_alias
        )