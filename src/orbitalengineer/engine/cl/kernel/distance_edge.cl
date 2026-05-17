#include "kernel/stride.clh"

__kernel void compute_edge_distance(
               const uint    N,
    __global   const float2* restrict position,
    __global   const float*  restrict radius,
    __global         float*  restrict distance_edge
) {
    GRID_STRIDE_INIT();
    GRID_STRIDE_IJ(
        float R = radius[j] + radius[i];
        float dist = distance(position[j], position[i]);
        distance_edge[IDX] = dist - R;
    );
}
