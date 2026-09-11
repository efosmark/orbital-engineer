#include "kernel/stride.clh"
#include "flags.clh"


inline float2 compute_gravitation(
    const float2 position_i,
    const float2 position_j,
    const float  mass_i,
    const float  mass_j
) {
    // Compute force (Newton's theory of universal gravitation)
    float2 dr = position_j - position_i;
    float dist = fast_length(dr);
    float2 mu = (float)(GRAV_CONSTANT) * mass_i * mass_j;
    return (mu / (dist*dist*dist)) * dr;
}

__kernel void compute_velocity(
             const uint    N,
             const float   dt,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       float2* restrict velocity,
    __global       float2* restrict velocity_intermediate,
    __global       float2* restrict force
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&FIXED_VELOCITY)) {
        if (lane == 0) velocity_intermediate[i] = velocity[i];
        return;
    }

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    GRID_STRIDE_IJ(
        if ((flags[j]&REMOVED)) continue;

        force[IDX] = compute_gravitation(position[i], position[j], mass[i], mass[j]);

        float2 dP = position[j] - position[i];
        float center_dist = fast_length(dP);

        float R = radius[i] + radius[j];
        float edge_dist = center_dist - R;

        float2 accel = force[IDX] * inv_mass_i * ((edge_dist < -EPS_DIST) ? -1.0f : 1.0f);
        dV_accum += accel * dt;
    );

    float2 wg_dV = FLOAT2_WG_REDUCE_ADD(dV_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + ((fast_length(wg_dV) <= DV_MAX) ? wg_dV : 0.0f);
    }
}