#include "kernel/stride.clh"
#include "flags.clh"

__kernel void apply_nudge(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       float2* restrict intermediate_position
) {
    GRID_STRIDE_INIT();

    float2 total_dP = 0;
    float inv_mass_i = 1.0 / mass[i];

    GRID_STRIDE_IJ(
        if ((flags[j]&REMOVED)) continue;
        float edge_dist = fast_length(position[j] - position[i]) - radius[i] - radius[j];
        if (edge_dist > 0) continue;
        
        float2 dP = position[j] - position[i];
        float2 r_norm = normalize(dP);
        float inv_mass_j = 1.0 / mass[j];
        float inv_mass_sum = (inv_mass_i + inv_mass_j);
        float k = edge_dist / inv_mass_sum;

        total_dP += r_norm * (k * inv_mass_i);    
    );

    float2 wg_dP = FLOAT2_WG_REDUCE_ADD(total_dP);
    if (lane == 0) intermediate_position[i] = position[i] + wg_dP;
}