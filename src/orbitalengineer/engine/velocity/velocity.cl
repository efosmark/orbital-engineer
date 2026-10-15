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

    GRID_STRIDE_IJ(
        if (j == i || (flags[j]&REMOVED)) continue;

        // Get (r + ϵ)**2
        float2 dr = position[j] - position_i;
        float r_squared = dot(dr, dr);
        float softened_r_squared = r_squared + (EPS_SOFTENING * EPS_SOFTENING);

        float inv_r = 1.0f / sqrt(softened_r_squared);
        float inv_r3 = pown(inv_r, 3);

        // Compute force (Newton's theory of universal gravitation)
        float mu = GRAV_CONSTANT * mass_i * mass[j];
        float2 F = mu * inv_r3 * dr;

        accel_accum += F * inv_mass_i;
        force[IDX] = F;
    );

    float2 wg_accel = FLOAT2_WG_REDUCE_ADD(accel_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + (wg_accel * dt);
    }
}
