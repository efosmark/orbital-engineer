#include "flags.clh"
#include "dtype.clh"

__kernel void compute_position(
             const float   dt,
    __global const uint*   restrict flags,
    __global const COMPLEX* restrict velocity,
    __global       COMPLEX* restrict position
) {
    uint i = get_global_id(0);
    if ((flags[i]&REMOVED)) return;
    position[i] += (flags[i]&FIXED_POSITION) ? 0.0f : velocity[i] * dt;
}