import numpy as np
import pyopencl as cl

from orbitalengineer.engine.cl.kernelizer import EventTracer, KernelBuilder
mf = cl.mem_flags


kernel_src = """
__kernel void global_reduce_min(
             float  min_val,
             float  max_val,
    __global float* input_data,
    __global uint *restrict indices
) {
    uint gid = get_global_id(0);

    uint min_idx = indices[gid];
    float wi_val = INFINITY;
    if (EPS_TIME <= input_data[min_idx] && input_data[min_idx] <= max_val) {
        wi_val = input_data[min_idx];
    }

    // Sub-group reduced value
    float sg_val = sub_group_reduce_min(wi_val);
    if (get_sub_group_local_id() != 0) {
        sg_val = INFINITY;
    }

    // Work-group minimum value
    float wg_val = work_group_reduce_min(sg_val);

    // Find the index of the value that matches the minimum
    uint earliest_id = work_group_reduce_min(
        (wg_val == wi_val) ? (uint)min_idx : INFINITY
    );

    // Assign the index to the minimum value in the work group
    if (earliest_id == min_idx) {
        indices[get_group_id(0)] = min_idx;
    }
}
"""


class GlobalReduceMin(KernelBuilder):
    
    kernel_name = "global_reduce_min"
    kernel_src = kernel_src

    def __call__(self, size:int, min_val:float, max_val:float, input_buffer: cl.Buffer, *, local_size=256, metric_alias:str|None=None):
        result_indices = np.arange(size, dtype=np.uint32)
        result_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_indices)
        
        while size > local_size:
            self.kernel(
                self.queue, (size,), (local_size,),
                np.float32(min_val),
                np.float32(max_val),
                input_buffer,
                result_buffer,
                metric_alias=metric_alias
            )
            size = int(np.ceil(size / local_size))
        
        # Final reduction
        self.kernel(
            self.queue, (size,), (size,),
        
            np.float32(min_val),
            np.float32(max_val),
            input_buffer,
            result_buffer,
            metric_alias=metric_alias
        )
        
        cl.enqueue_copy(self.queue, result_indices[:1], result_buffer).wait()
        result = result_indices[0]
        return result
    

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
    
    N = 10000
    rng = np.random.default_rng()
    values = np.array([rng.uniform(1, 1000) for i in range(int(N))], dtype=np.float32)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    global_reduce_min = GlobalReduceMin(ctx, queue, EventTracer(None))  

    input_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,  hostbuf=values)
    size = values.size
    result_indices = np.arange(values.size, dtype=np.uint32)
    num_runs = 1
    
    # Profile the parallel runner
    with profile_func():
        for i in range(num_runs):
            global_reduce_min(size, 0, np.inf, input_buffer, local_size=256)
    
    # Profile the builtin function
    with profile_func():
        for i in range(num_runs):
            x = min(values)
     
    # Compare values
    indices = global_reduce_min(size, -np.inf, np.inf, input_buffer, local_size=256)
    print(values[indices], min(values))