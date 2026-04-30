import pyopencl as cl
import numpy as np

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags

DEFAULT_LX = 16

kernel_src = """
inline uint triu_index(uint i, uint j, uint N) {
    return i * (N - 1u) - (i * (i - 1u)) / 2u + (j - i - 1u);
}

__kernel void pairwise_reduce(
             const uint    N,
    __global const float2* pair_ij,
    __global const float2* pair_ji,
    __global       float2* result
) {
    uint node = get_group_id(0);
    if (node >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    float2 accum = (float2)(0.0f, 0.0f);

    // contiguous half: pairs (i, j), j > i
    uint count_upper = N - 1u - node;
    if (count_upper > 0) {
        uint start = triu_index(node, node + 1u, N);
        for (uint off = lane; off < count_upper; off += Lx) {
            accum += pair_ij[start + off];
        }
    }

    // scattered half: pairs (j, i), j < i
    for (uint j = lane; j < node; j += Lx) {
        uint k = triu_index(j, node, N);
        accum += pair_ji[k];
    }
    
    float real = accum.x;
    float imag = accum.y;

    float sx = work_group_reduce_add(real);
    float sy = work_group_reduce_add(imag);

    if (lane == 0) {
        result[node] += (float2)(sx, sy);
    }
}
"""

class PairwiseNodeReduce(KernelBuilder):
    kernel_name = "pairwise_reduce"
    kernel_src = kernel_src
    
    def __call__(
        self,
        N:int,
        pair_ij:cl.Buffer,
        pair_ji:cl.Buffer,
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
            pair_ij,
            pair_ji,
            output_buffer,
            
            metric_alias=metric_alias
        )


kernel_src = """

inline uint triu_index(uint i, uint j, uint N) {
    return i * (N - 1u) - (i*(i - 1u)) / 2u + (j - i - 1);
}

__kernel void pairwise_reduce_direct(
    __global const float2* pair_ij,
    __global const float2* pair_ji,
    __global float2 *restrict result
) {
    uint N  = get_global_size(0);
    uint i = get_global_id(0);

    float2 accum = (float2) (0, 0);

    // contiguous half
    uint start = triu_index(i, i + 1, N);
    for (uint idx = start; idx < start + (N - 1 - i); ++idx) {
        accum += pair_ij[idx];
    }

    // scattered half
    for (uint j = 0; j < i; ++j) {
        uint k = triu_index(j, i, N);
        accum += pair_ji[k];
    }

    result[i] += accum;
}
"""

class PairwiseNodeReduceDirect(KernelBuilder):
    kernel_name = "pairwise_reduce_direct"
    kernel_src = kernel_src
    
    def __call__(self, N:int, pair_ij:cl.Buffer, pair_ji:cl.Buffer, output_buffer:cl.Buffer, *, local_size:int=256, increment=True, metric_alias:str|None=None):
        Lx = int(local_size)
        return self.kernel(
            self.queue,
            (N, ), # global work size
            None,
            
            # Args
            pair_ij,
            pair_ji,
            output_buffer,
            
            metric_alias=metric_alias
        )