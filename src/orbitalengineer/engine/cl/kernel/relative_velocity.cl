#include "kernel/complex.clh" // needed for creal, cmul, cconj
#include "kernel/stride.clh"

inline float relative_speed_along_normal(
    const uint i,
    const uint j,
    const float2 r_norm,
    const float2* velocity
) {
    float2 vr = velocity[i] - velocity[j]; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}


__kernel void compute_relative_velocity(
               const uint             N,
    __global   const float2* restrict position,
    __global   const float2* restrict velocity,
    __global   const float*  restrict radius,
    __global         float*  restrict relative_velocity
) {
    GRID_STRIDE_INIT();
    GRID_STRIDE_IJ(
        float R = radius[j] + radius[i];
        float2 dP = position[j] - position[i];

        float2 r_norm = normalize(dP);
        relative_velocity[row_start + j] = relative_speed_along_normal(i, j, -r_norm, velocity);
    );
}