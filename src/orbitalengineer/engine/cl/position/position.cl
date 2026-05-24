#include "flags.clh"

__kernel void compute_position(
             const float   dt,
    __global const uint*   restrict flags,
    __global const float2* restrict velocity,
    __global       float2* restrict position
) {
    uint i = get_global_id(0);
    if ((flags[i]&REMOVED)) return;
    position[i] += velocity[i] * dt;
}