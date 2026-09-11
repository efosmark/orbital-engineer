#include "kernel/stride.clh"
#include "flags.clh"

inline float2 compute_time_of_impact(const float2 dV, const float2 dP, const float R) {
    // Coefficients
    float a = dot(dV, dV);
    float b = 2.0f * dot(dP, dV);
    float c = dot(dP, dP) - R * R;

    // Already inside or touching the interaction radius
    if (c < 0.0f) return 0.0f;

    // No relative motion -> never enters in the future
    const float eps = 1e-12f;
    //if (a <= eps) return INFINITY;

    // Discriminant
    float D = b * b - 4.0f * a * c;

    //if (!isfinite(D))
    //    return INFINITY;

    float sqrtD = sqrt(D);
    float inv2a = 0.5f / a;
    float t1 = (-b - sqrtD) * inv2a;
    float t2 = (-b + sqrtD) * inv2a;

    return (float2)(t1, t2);
}

__kernel void interaction_time(
             const uint    N,
             const float   dt_step,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict radius,
    __global       float2* restrict dt_until_collision,
    __global       float*  restrict min_dt_per_body
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    float min_impact = dt_step;
    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;

        float R = radius[j] + radius[i];
        float2 dV = velocity[j] - velocity[i];
        float2 dP = position[j] - position[i];

        float2 dt_until_collision_ij = compute_time_of_impact(dV, dP, R);
        dt_until_collision[IDX] = dt_until_collision_ij;

        float t1 = dt_until_collision_ij.x;
        float t2 = dt_until_collision_ij.y;
        float curr_min = (t1 > 0.0f) ? t1 : ((t2 > 0.0f) ? t2 : dt_step);
        min_impact = fmin(min_impact, curr_min);
    );

    float wg_min_impact = work_group_reduce_min(min_impact);
    if (lane == 0)
        min_dt_per_body[i] = wg_min_impact;
}