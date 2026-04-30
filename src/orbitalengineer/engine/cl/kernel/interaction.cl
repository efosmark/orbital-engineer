#include "kernel/complex.clh"

#define IN_NONE 0
#define IN_APPROACHING 1
#define IN_COLLISION 2
#define IN_DEPARTING 3
#define IN_ORBITING 4
#define IN_ORBITED 5
#define IN_OVERLAPPING 6

inline uint triu_index(uint i, uint j, uint N) {
    // NOTE: `i` should always be less than `j`
    return i * (N - 1u) - (i * (i - 1u)) / 2u + (j - i - 1u);
}

inline float2 clamp_scale(float2 z, float limit) {
    float mag = length(z);
    float scale = (mag > 0.0f) ? fmin(1.0f, limit / mag) : 1.0f;
    return z * scale;
}

inline float compute_time_of_impact(
    const cfloat dV,
    const cfloat dP,
    const float  R
) {
    float a = dot(dV, dV);
    float b = 2.0f * dot(dP, dV);
    float c = dot(dP, dP) - R * R;

    // Already inside or touching the interaction radius
    if (c <= 0.0f) {
        return 0.0f;
    }

    // No relative motion -> never enters in the future
    const float eps = 1e-12f;
    if (a <= eps) {
        return INFINITY;
    }

    float D = b * b - 4.0f * a * c;
    if (a <= eps || D < 0.0f || !isfinite(D))
        return INFINITY;

    float sqrtD = sqrt(D);

    float inv2a = 0.5f / a;
    float t1 = (-b - sqrtD) * inv2a;
    float t2 = (-b + sqrtD) * inv2a;

    return (t1 >= 0.0f) 
           ? t1
           : ((t1 >= 0.0f) ? t2 : INFINITY);
}

__kernel void compute_interactions(
               const uint    N,
    __global   const cfloat* restrict position,
    __global   const cfloat* restrict velocity,
    __global   const float*  restrict radius,
    __global         float*  restrict impact_times,
    __global         float*  restrict min_impact_time
) {
    uint i = get_group_id(0);
    if (i >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx = get_local_size(0);

    float min_impact = INFINITY;
    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        if (j == i) continue;
        
        float R = radius[j] + radius[i];
        cfloat dV = velocity[j] - velocity[i];
        cfloat dP = position[j] - position[i];

        float toi_ij = compute_time_of_impact(dV, dP, R);
        impact_times[row_start + j] = toi_ij;
        min_impact = fmin(min_impact, toi_ij);
    }

    float wg_min_impact = work_group_reduce_min(min_impact);
    if (lane == 0) { 
        min_impact_time[i] = wg_min_impact;
    }
}
