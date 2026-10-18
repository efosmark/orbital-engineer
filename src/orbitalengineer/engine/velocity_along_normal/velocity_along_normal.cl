#include "kernel/complex.clh"
#include "kernel/stride.clh"
#include "flags.clh"

inline REAL relative_speed_along_normal(
    const COMPLEX velocity_i,
    const COMPLEX velocity_j,
    const COMPLEX r_norm
) {
    COMPLEX vr = velocity_i - velocity_j; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}

__kernel void velocity_along_normal(
             const uint     N_bodies_alloc,
             const uint     N_bodies_valid,
    __global const uint*    restrict flags,
    __global const COMPLEX* restrict position,
    __global const COMPLEX* restrict velocity,
    __global       REAL*    restrict out
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        COMPLEX dP = position[j] - position[i];
        COMPLEX r_norm = normalize(dP);
        out[IDX] = relative_speed_along_normal(velocity[i], velocity[j], -r_norm);
    );
}
