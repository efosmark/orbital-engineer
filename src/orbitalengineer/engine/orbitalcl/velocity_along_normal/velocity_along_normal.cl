#include "kernel/complex.clh"
#include "kernel/stride.clh"
#include "flags.clh"

inline float relative_speed_along_normal(
    const float2 velocity_i,
    const float2 velocity_j,
    const float2 r_norm
) {
    float2 vr = velocity_i - velocity_j; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}

__kernel void velocity_along_normal(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global       float*  restrict out
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        float2 dP = position[j] - position[i];
        float2 r_norm = normalize(dP);
        out[IDX] = relative_speed_along_normal(velocity[i], velocity[j], -r_norm);
    );
}
