#include "kernel/stride.clh"
#include "flags.clh"

inline double get_kinetic_energy(
    double mass,
    double2 velocity
) {
    return 0.5 * mass * dot(velocity, velocity);
}

inline double get_gravitational_potential_energy(double mass_i, double mass_j, double2 position_i, double2 position_j) {
    double mu = (double)GRAV_CONSTANT * mass_i * mass_j;

    double2 dr = position_i - position_j;
    double r2 = dot(dr, dr);

    double eps = (double)EPS_SOFTENING;
    double softened_r2 = r2 + eps * eps;

    return -mu / sqrt(softened_r2);
}

__kernel void compute_energy(
             const uint     N_bodies_alloc,
             const uint     N_bodies_valid,
    __global const uint*    flags,
    __global const float2*  position,
    __global const float2*  velocity,
    __global const float*   mass,
    __global const float*   radius,
    __global       double*  kinetic_energy,
    __global       double*  grav_potential_energy
) {
    GRID_STRIDE_INIT();
    if (i >= N_bodies_valid) return;
    if (flags[i]&REMOVED) return;

    double mass_i = convert_double(mass[i]);
    double2 position_i = convert_double2(position[i]);

    double potential_energy = 0.0f;
    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        if (i == j) continue;

        double mass_j = convert_double(mass[j]);
        double2 position_j = convert_double2(position[j]);

        potential_energy += get_gravitational_potential_energy(mass_i, mass_j, position_i, position_j);
    );
    
    double total_potential_energy = work_group_reduce_add(potential_energy);
    if (lane == 0) {
        double2 velocity_i = convert_double2(velocity[i]);
        kinetic_energy[i] = get_kinetic_energy(mass_i, velocity_i);
        grav_potential_energy[i] = 0.5f * total_potential_energy;
    }
}