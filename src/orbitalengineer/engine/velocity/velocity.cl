#include "kernel/stride.clh"
#include "flags.clh"
#include "dtype.clh"

__kernel void compute_velocity(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
             const float   dt,
    __global const uint*   flags,
    __global const COMPLEX* position,
    __global const float*  mass,
    __global const float*  radius,
    __global const COMPLEX* velocity,
    __global const bool*   is_touching,
    __global       COMPLEX* velocity_intermediate,
    __global       COMPLEX* force
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&FIXED_VELOCITY)) {
       if (lane == 0) velocity_intermediate[i] = velocity[i];
       return;
    }

    COMPLEX accel_accum = (COMPLEX)(0.0f, 0.0f);
    REAL    inv_mass_i = 1.0f / mass[i];

    COMPLEX position_i = position[i];
    REAL    radius_i = radius[i];
    REAL    mass_i = mass[i];

    GRID_STRIDE_IJ(
        if (j == i || (flags[j]&REMOVED)) continue;

        // Get (r + ϵ)**2
        COMPLEX dr = position[j] - position_i;
        REAL    r_squared = dot(dr, dr);
        REAL    softened_r_squared = r_squared + (SOFTENING_LENGTH * SOFTENING_LENGTH);

        REAL inv_r = 1.0f / sqrt(softened_r_squared);
        REAL inv_r3 = pown(inv_r, 3);

        // Compute force (Newton's theory of universal gravitation)
        REAL    mu = GRAV_CONSTANT * mass_i * mass[j];
        COMPLEX F = mu * inv_r3 * dr;

        accel_accum += F * inv_mass_i;
        force[IDX] = F;
    );

    COMPLEX wg_accel = COMPLEX_WG_REDUCE_ADD(COMPLEX, accel_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + (wg_accel * dt);
    }
}
