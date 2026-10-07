#include "kernel/stride.clh"
#include "flags.clh"

__kernel void compute_velocity(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
             const float   dt,
    __global const uint*   flags,
    __global const float2* position,
    __global const float*  mass,
    __global const float*  radius,
    __global const float2* velocity,
    __global const bool*   is_touching,
    __global       float2* velocity_intermediate,
    __global       float2* force
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&FIXED_VELOCITY)) {
       if (lane == 0) velocity_intermediate[i] = velocity[i];
       return;
    }

    float2 accel_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    float2 position_i = position[i];
    float radius_i = radius[i];
    float mass_i = mass[i];

    uint row_start = i * N_bodies_alloc;
    for (uint j = lane; j < N_bodies_valid; j += Lx) {
        if (j == i || (flags[j]&REMOVED)) continue;
        uint IDX = row_start + j;

        float2 dr = position[j] - position_i;
        float dist = fast_length(dr);
        
        // Compute force (Newton's theory of universal gravitation)
        float2 mu = (float)(GRAV_CONSTANT) * mass_i * mass[j];
        float2 F = (mu / (dist*dist*dist)) * dr;

        float edge_distance = dist - radius_i - radius[j];
        accel_accum += F * inv_mass_i * ((edge_distance > 0) ? 1 : -1);

        force[IDX] = F;
    };

    float2 wg_accel = FLOAT2_WG_REDUCE_ADD(accel_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + (wg_accel * dt);// + ((fast_length(wg_accel) <= DV_MAX) ? wg_accel : 0.0f);
    }
}
