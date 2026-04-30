import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags

DEFAULT_LX = 32

kernel_src = """
inline uint triu_index(uint i, uint j, uint N) {
    return i * (N - 1u) - (i * (i - 1u)) / 2u + (j - i - 1u);
}

__kernel void pairwise_min_reduce(
             const uint    N,
             const float   min_value,
    __global const float*  pair,
    __global       float*  result
) {
    uint node = get_group_id(0);
    if (node >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    uint start = triu_index(node, node + 1u, N);
    float res = INFINITY;

    // contiguous half: pairs (i, j), j > i
    uint count_upper = N - 1u - node;
    if (count_upper > 0) {
        for (uint off = lane; off < count_upper; off += Lx) {
            float val = pair[start + off];
            if (val >= min_value)
                res = fmin(res, val);
        }
    }

    // scattered half: pairs (j, i), j < i
    for (uint j = lane; j < node; j += Lx) {
        uint k = triu_index(j, node, N);
        float val = pair[k];
        if (val >= min_value)
            res = fmin(res, val);
    }

    float wg_result = work_group_reduce_min(res);
    if (lane == 0) {
        result[node] = wg_result;
    }
}
"""

class PairwiseMinReduce(KernelBuilder):
    kernel_name = "pairwise_min_reduce"
    kernel_src = kernel_src
    
    def __call__(
        self,
        N:int,
        min_value: np.float32,
        pair:cl.Buffer,
        output_buffer:cl.Buffer,
        *,
        local_size:int=DEFAULT_LX,
        metric_alias:str|None=None
    ):
        Lx = min(N, int(local_size))
        return self.kernel(
            self.queue,
            (N * Lx, ), # global work size
            (Lx, ),     # local work size
            
            # Args
            np.uint32(N),
            min_value,
            pair,
            output_buffer,
            
            metric_alias=metric_alias
        )
