#include "kernel/stride.clh"
#include "flags.clh"
#include "dtype.clh"

inline float2 compute_time_of_impact(const COMPLEX dV, const COMPLEX dP, const float R) {
    // Coefficients
    REAL a = dot(dV, dV);
    REAL b = 2.0f * dot(dP, dV);
    REAL c = dot(dP, dP) - R * R;

    // Already inside or touching the interaction radius
    if (c < 0.0f) return 0.0f;

    // No relative motion -> never enters in the future
    const float eps = 1e-12f;
    //if (a <= eps) return INFINITY;

    // Discriminant
    REAL D = b * b - 4.0f * a * c;

    //if (!isfinite(D))
    //    return INFINITY;

    REAL sqrtD = sqrt(D);
    REAL inv2a = 0.5f / a;
    float t1 = (-b - sqrtD) * inv2a;
    float t2 = (-b + sqrtD) * inv2a;

    return (float2)(t1, t2);
}

__kernel void compute_interaction(
             const uint     N_bodies_alloc,
             const uint     N_bodies_valid,
             const float    dt_step,
    __global const uint*    flags,
    __global const COMPLEX* position,
    __global const COMPLEX* velocity,
    __global const float*   radius,
    __global       float2*  dt_until_collision,
    __global       float*   min_dt_per_body
) {
    GRID_STRIDE_INIT();
    if (lane == 0) {
        min_dt_per_body[i] = INFINITY;
    }

    if (flags[i]&REMOVED) return;

    float min_impact = dt_step;
    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;

        float R = radius[j] + radius[i];
        COMPLEX dV = velocity[j] - velocity[i];
        COMPLEX dP = position[j] - position[i];

        float2 dt_until_collision_ij = compute_time_of_impact(dV, dP, R);
        dt_until_collision[IDX] = dt_until_collision_ij;

        float t1 = dt_until_collision_ij.x;
        float t2 = dt_until_collision_ij.y;
        float curr_min = (t1 > 0.0f) ? t1 : ((t2 > 0.0f) ? t2 : dt_step);
        min_impact = fmin(min_impact, curr_min);
    );

    float wg_min_impact = work_group_reduce_min(min_impact);
    if (lane == 0) {
        min_dt_per_body[i] = wg_min_impact;
    }
}