#include "kernel/stride.clh"

__kernel void apply_nudge(
               const uint    N,
    __global   const float2* position,
    __global   const float*  mass,
    __global   const float*  distance_edge,
    __global         float2* intermediate_position
) {
    GRID_STRIDE_INIT();

    float2 total_dP = 0;
    float inv_mass_i = 1.0 / mass[i];

    GRID_STRIDE_IJ(
        if (distance_edge[IDX] > 0) continue;
        
        float2 dP = position[j] - position[i];
        float2 r_norm = normalize(dP);
        float inv_mass_j = 1.0 / mass[j];
        float inv_mass_sum = (inv_mass_i + inv_mass_j);
        float k = distance_edge[IDX] / inv_mass_sum;

        total_dP += r_norm * (k * inv_mass_i);    
    );

    float2 wg_dP = FLOAT2_WG_REDUCE_ADD(total_dP);
    if (lane == 0) intermediate_position[i] = position[i] + wg_dP;
}