#include "kernel/stride.clh"

#ifndef G
#define G 1.0
#endif


inline float2 compute_gravitation(
    const float2 position_i,
    const float2 position_j,
    const float  mass_i,
    const float  mass_j
) {
    // Compute force (Newton's theory of universal gravitation)
    float2 dr = position_j - position_i;
    float dist = fast_length(dr);
    float2 mu = (float)(G) * mass_i * mass_j;
    return (mu / (dist*dist*dist)) * dr;
}

__kernel void compute_velocity(
               const uint    N,
               const float   dt,
    __global   const float2* restrict position,
    __global   const float*  restrict mass,
    __global         float2* restrict velocity
) {
    GRID_STRIDE_INIT();

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f/ mass[i];

    GRID_STRIDE_IJ(
        if (mass[j] == 0) {
            continue;
        }

        float2 force = compute_gravitation(position[i], position[j], mass[i], mass[j]);

        // Apply the acceleration
        float2 acceleration = force * inv_mass_i;
        dV_accum += acceleration * dt;
    );
  
    float2 wg_dV = FLOAT2_WG_REDUCE_ADD(dV_accum);
    if (lane == 0) { 
        velocity[i] += wg_dV;
    }
}