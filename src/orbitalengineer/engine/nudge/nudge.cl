#include "kernel/stride.clh"
#include "flags.clh"
#include "dtype.clh"

__kernel void apply_nudge(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
    __global const uint*   restrict flags,
    __global const COMPLEX* restrict position,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       COMPLEX* restrict intermediate_position
) {
    GRID_STRIDE_INIT();

    COMPLEX total_dP = 0;
    float inv_mass_i = 1.0 / mass[i];

    GRID_STRIDE_IJ(
        if ((flags[j]&REMOVED)) continue;
        float edge_dist = length(position[j] - position[i]) - radius[i] - radius[j];
        if (edge_dist > 0) continue;
        
        COMPLEX dP = position[j] - position[i];
        COMPLEX r_norm = normalize(dP);
        float inv_mass_j = 1.0 / mass[j];
        float inv_mass_sum = (inv_mass_i + inv_mass_j);
        float k = edge_dist / inv_mass_sum;

        total_dP += r_norm * (k * inv_mass_i);
    );

    COMPLEX wg_dP = COMPLEX_WG_REDUCE_ADD(COMPLEX, total_dP);
    if (lane == 0) intermediate_position[i] = position[i] + wg_dP;
}