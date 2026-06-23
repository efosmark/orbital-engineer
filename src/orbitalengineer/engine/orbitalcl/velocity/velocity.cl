#include "kernel/stride.clh"
#include "flags.clh"

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
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const float*  restrict distance_edge,
    __global       float2* restrict acceleration,
    __global       float2* restrict velocity
) {
    GRID_STRIDE_INIT();
    if ((flags[i]&FIXED_VELOCITY) || (flags[i]&REMOVED)) return;

    float2 A = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    GRID_STRIDE_IJ(
        if ((flags[j]&REMOVED)) continue;
        float2 force = compute_gravitation(position[i], position[j], mass[i], mass[j]);
        float repel = (flags[i]&REPEL_ON_OVERLAP) ? -1.0f : 0;

        float2 accel = force * inv_mass_i * ((distance_edge[IDX] < 0) ? repel : 1.0f);
        acceleration[IDX] = accel;
        A += accel;
    );
  
    float2 wg_A = FLOAT2_WG_REDUCE_ADD(A);
    if (lane == 0) velocity[i] += wg_A * dt;
}