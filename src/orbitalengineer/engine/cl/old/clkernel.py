import numpy as np
from numpy.typing import NDArray

import pyopencl as cl
mf = cl.mem_flags


kernel_src = r"""
__kernel void global_reduce_min(
    __global const float*    input_data,
    __global uint *restrict  indices
) {
    uint gid = get_global_id(0);

    uint min_idx = indices[gid];
    float min_val = input_data[min_idx];

    // Sub-group minimum value
    float sg_min_val = sub_group_reduce_min(min_val);
    if (get_sub_group_local_id() != 0) {
        sg_min_val = INFINITY;
    }

    // Work-group minimum value
    float wg_min_val = work_group_reduce_min(sg_min_val);

    // Find the index of the value that matches the minimum
    uint earliest_id = work_group_reduce_min(
        (wg_min_val == min_val) ? (uint)min_idx : INFINITY
    );

    // Assign the index to the minimum value in the work group
    if (earliest_id == min_idx) {
        indices[get_group_id(0)] = min_idx;
    }
}
"""


class CLKernel:
    
    def __init__(self, ctx:cl.Context, queue:cl.CommandQueue):
        self.ctx = ctx
        self.queue = queue
        self.prg = cl.Program(self.ctx, kernel_src).build(
            options=['-cl-std=CL2.0']
        )
        self.kernel = cl.Kernel(self.prg, 'global_reduce_min')

    


    def __call__(self, size:int, input_buffer: cl.Buffer, *, local_size=256):
        result_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_indices)
        
        while size > local_size:
            self.kernel(self.queue, (size,), (local_size,), input_buffer, result_buffer)
            size = int(np.ceil(size / local_size))
        
        # Final reduction
        self.kernel(self.queue, (size,), (size,), input_buffer, result_buffer)
        
        cl.enqueue_copy(queue, result_indices[:1], result_buffer).wait()
        return result_indices[0]
    

if __name__ == "__main__":  
    from contextlib import contextmanager
    import time
    
    @contextmanager
    def profile_func():
        start = time.perf_counter()
        yield 
        duration_ns = ((time.perf_counter() - start)) * 1000.0 * 1000.0
        print(f"Duration: {duration_ns:.2f} ns")
    
    import os
    os.environ['PYOPENCL_CTX'] = '0'
    
    N = 50000
    rng = np.random.default_rng()
    values = np.array([rng.uniform(1, 1000) for i in range(int(N))], dtype=np.float32)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    global_reduce_min = GlobalReduceMin(ctx, queue)  

    input_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,  hostbuf=values)
    size = values.size
    result_indices = np.arange(values.size, dtype=np.uint32)
    num_runs = 1
    
    # Profile the parallel runner
    with profile_func():
        for i in range(num_runs):
            global_reduce_min(size, input_buffer, local_size=256)
    
    # Profile the builtin function
    with profile_func():
        for i in range(num_runs):
            x = min(values)
    
    # Compare values
    indices = global_reduce_min(size, input_buffer, local_size=256)
    print(values[indices], min(values))