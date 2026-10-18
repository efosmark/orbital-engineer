#include "kernel/stride.clh"
#include "flags.clh"
#include "dtype.clh"

inline double get_kinetic_energy(float mass, COMPLEX velocity) {
    return 0.5 * mass * convert_double(dot(velocity, velocity));
}

inline REAL get_gravitational_potential_energy(float mass_i, float mass_j, COMPLEX position_i, COMPLEX position_j) {
    REAL mu = (REAL)GRAV_CONSTANT * mass_i * mass_j;

    COMPLEX dr = position_i - position_j;
    REAL    r2 = dot(dr, dr);

    REAL eps = (REAL)SOFTENING_LENGTH;
    REAL softened_r2 = r2 + eps * eps;

    return -mu / sqrt(softened_r2);
}

__kernel void compute_energy(
             const uint     N_bodies_alloc,
             const uint     N_bodies_valid,
    __global const uint*    flags,
    __global const COMPLEX* position,
    __global const COMPLEX* velocity,
    __global const float*   mass,
    __global const float*   radius,
    __global       double*  kinetic_energy,
    __global       double*  grav_potential_energy
) {
    GRID_STRIDE_INIT();
    if (i >= N_bodies_valid) return;
    if (flags[i]&REMOVED) return;

    float mass_i = mass[i];
    COMPLEX position_i = position[i];

    double potential_energy = 0.0f;
    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        if (i == j) continue;

        float mass_j = mass[j];
        COMPLEX position_j = position[j];
        potential_energy += get_gravitational_potential_energy(mass_i, mass_j, position_i, position_j);
    );
    
    double total_potential_energy = work_group_reduce_add(potential_energy);
    if (lane == 0) {
        COMPLEX velocity_i = velocity[i];
        kinetic_energy[i] = get_kinetic_energy(mass_i, velocity_i);
        grav_potential_energy[i] = 0.5f * total_potential_energy;
    }
}