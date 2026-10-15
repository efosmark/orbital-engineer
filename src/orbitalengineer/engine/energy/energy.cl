#include "kernel/stride.clh"
#include "flags.clh"

inline float get_kinetic_energy(float mass, float2 velocity) {
    return 0.5 * mass * pow(fast_length(velocity), 2);
}

inline float get_gravitational_potential_energy(
    float  mass_i,
    float  mass_j,
    float2 position_i,
    float2 position_j
) {
    float mu = GRAV_CONSTANT * mass_i * mass_j;
    float r = fast_length(position_i - position_j);
    return -mu / (r + EPS_SOFTENING);
}

__kernel void compute_energy(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
    __global const uint*   flags,
    __global const float2* position,
    __global const float2* velocity,
    __global const float*  mass,
    __global const float*  radius,
    __global       float*  kinetic_energy,
    __global       float*  grav_potential_energy
) {
    GRID_STRIDE_INIT();
    if (i >= N_bodies_valid) return;
    if (flags[i]&REMOVED) return;

    if (lane == 0) {
        kinetic_energy[i] = get_kinetic_energy(mass[i], velocity[i]);
    }

    float potential_energy = 0.0f;    
    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        if (i == j) continue;
        potential_energy += get_gravitational_potential_energy(mass[i], mass[j], position[i], position[j]);
    );
    
    float total_potential_energy = work_group_reduce_add(potential_energy);
    if (lane == 0) {
        grav_potential_energy[i] = 0.5f * total_potential_energy;
    }
}