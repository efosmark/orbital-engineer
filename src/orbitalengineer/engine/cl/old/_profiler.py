import statistics
import numpy as np
import pyopencl as cl

from orbitalengineer.engine.cl.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
__kernel void mem_bandwidth_test(__global const float *src,
                                 __global float *dst)
{
    size_t gid = get_global_id(0);
    dst[gid] = src[gid];
}
"""


class MemBandwidthTest(KernelBuilder):
    
    kernel_name = "mem_bandwidth_test"
    kernel_src = kernel_src

    def __call__(self, size:int):
        
        src_host = np.ndarray(size, dtype=np.float32)
        src_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_host)
        
        dest_host = np.ndarray(size, dtype=np.float32)
        dest_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=dest_host)
        
        evt = self.kernel(self.queue, (size,), None, src_cl, dest_cl)
        evt.wait()
        ns = evt.profile.end - evt.profile.start
        secs = ns * 1e-9
        
        bytes_per_elem = 8
        total_bytes = N * bytes_per_elem
        gb = total_bytes / 1e9

        secs = (evt.profile.end - evt.profile.start) * 1e-9
        gbps = gb / secs
        #print(f"{secs:.6} seconds")
        #print(f"bandwidth: {gbps:.2f} GB/s")
        return gbps

if __name__ == "__main__":  
    from contextlib import contextmanager
    import time
    

    import os
    os.environ['PYOPENCL_CTX'] = '0'
    
    N = int(1e6)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mem_bandwidth_test = MemBandwidthTest(ctx, queue)
    
    for n in [7, 8]:
        N = 10 ** n
        avg = statistics.mean([mem_bandwidth_test(int(N)) for i in range(100)])
        print(f"N = 1e{n}   gbps = {avg:.2f}")