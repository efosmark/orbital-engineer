#include "kernel/stride.clh"
#include "flags.clh"

__kernel void edge_distance(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict radius,
    __global       float*  restrict out,
    __global       bool*   restrict is_touching
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        float R = radius[j] + radius[i];
        float2 dV = velocity[j] - velocity[i];
        float2 dP = position[j] - position[i];
        out[IDX] = fast_length(dP) - R;
        is_touching[IDX] = out[IDX] <= EPS_DIST
    );
}