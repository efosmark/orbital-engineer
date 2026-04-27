#include "kernel/complex.clh" // needed for creal, cmul, cconj

inline float relative_speed_along_normal(
    const uint i,
    const uint j,
    const float2 r_norm,
    const float2* velocity
) {
    float2 vr = velocity[i] - velocity[j]; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}


__kernel void compute_velocity_relative(
               const uint             N,
    __global   const float2* restrict position,
    __global   const float2* restrict velocity,
    __global   const float*  restrict radius,
    __global         float*  restrict v_rel
) {
    uint i = get_group_id(0);
    if (i >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        float R = radius[j] + radius[i];
        float2 dP = position[j] - position[i];

        float2 r_norm = normalize(dP);
        v_rel[row_start + j] = relative_speed_along_normal(i, j, -r_norm, velocity);
    }
}