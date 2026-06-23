#include "kernel/stride.clh"

__kernel void compute_hill_radius(
             const uint    N,
    __global const float2* restrict position,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       float*  restrict hill_radius,
    __global       float*  restrict minimum_hill_radius
) {
    GRID_STRIDE_INIT();

    float minimum_hill_radius_i = INFINITY;
    GRID_STRIDE_IJ(
        
        float dist = distance(position[j], position[i]);
        
        //hill_radius[IDX] = dist * cbrt(mass[i] / (3.0f * (mass[i] + mass[j])));
        
        // 
        hill_radius[IDX] = dist * pow(min(mass[i], mass[j]) / max(mass[i], mass[j]), 2.0f/5.0f);
        

        minimum_hill_radius_i = (hill_radius[IDX] > 0)
                              ? fmin(minimum_hill_radius_i, hill_radius[IDX])
                              : minimum_hill_radius_i;
    );

    float wg_minimum_hill_radius = work_group_reduce_min(minimum_hill_radius_i);
    if (lane == 0)
        minimum_hill_radius[i] = wg_minimum_hill_radius;
}
