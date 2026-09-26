#include "kernel/stride.clh"
#include "flags.clh"

typedef struct {
    const uint idx;
    const uint i;
    const uint j;
} Pair;

inline uint triu_index(uint i, uint j, uint N) {
    return i * (N - 1u) - (i * (i - 1u)) / 2u + (j - i - 1u);
}

inline float2 compute_gravitation(
    const float2 position_i,
    const float2 position_j,
    const float  mass_i,
    const float  mass_j
) {
    // Compute force (Newton's theory of universal gravitation)
    float2 dr = position_j - position_i;
    float dist = fast_length(dr);
    float2 mu = (float)(GRAV_CONSTANT) * mass_i * mass_j;
    return (mu / (dist*dist*dist)) * dr;
}

__kernel void compute_velocity(
             const uint    N,
             const float   dt,
    __global const uint*   flags,
    __global const float2* position,
    __global const float*  mass,
    __global const float*  radius,
    __global const float2* velocity,
    __global const bool*   is_touching,
    __global       float2* velocity_intermediate,
    __global       float2* force
) {
    GRID_STRIDE_INIT();

    //if ((flags[i]&REMOVED) || (flags[i]&FIXED_VELOCITY)) {
    //    if (lane == 0) velocity_intermediate[i] = velocity[i];
    //    return;
    //}

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    float2 position_i = position[i];
    float radius_i = radius[i];
    float mass_i = mass[i];

    uint row_start = i * N;
    bool* touching = &is_touching[row_start];
    float2* F = &force[row_start];

    for (uint j = lane; j < N; j += Lx) {
        if (j == i) continue;
        uint IDX = row_start + j;
        force[IDX] = compute_gravitation(position_i, position[j], mass_i, mass[j]) ;//* ((flags[j]&REMOVED)||(j == i));
        
        float2 dr = position[j] - position_i;
        float edge_distance = fast_length(dr) - radius_i - radius[j];
        float2 accel = force[IDX] * inv_mass_i ;//* ((edge_distance >= 0) ? 1 : -1);
        dV_accum += accel * dt;
    };

    float2 wg_dV = FLOAT2_WG_REDUCE_ADD(dV_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + wg_dV;// + ((fast_length(wg_dV) <= DV_MAX) ? wg_dV : 0.0f);
    }
}


__kernel void compute_velocity_pairwise(
             const uint    N,
             const float   dt,
    __global const Pair*   pairs,
    __global const uint*   flags,
    __global const float2* position,
    __global const float*  mass,
    __global const float*  radius,
    __global const float2* velocity,
    __global       float2* velocity_updates
) {
    uint g0 = get_global_id(0);

    Pair pair = pairs[g0];
    uint i = pair.i;
    uint j = pair.j;

    uint idx_i = (i * N) + j;
    uint idx_j = (j * N) + i;

    if ((flags[i]&REMOVED) || (flags[j]&REMOVED)) {
        velocity_updates[idx_i] = 0.0f;
        velocity_updates[idx_j] = 0.0f;
        return;
    }

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];
    float inv_mass_j = 1.0f / mass[j];

    float2 F = compute_gravitation(position[i], position[j], mass[i], mass[j]);

    float2 accel_i = F * inv_mass_i;
    float2 accel_j = -F * inv_mass_j;

    velocity_updates[idx_i] = accel_i * dt;
    velocity_updates[idx_j] = accel_j * dt;
}


__kernel void apply_velocity_updates(
             const uint    N,
    __global const float2* velocity,
    __global const float2* velocity_updates,
    __global       float2* velocity_intermediate
) {
    float2 dV_accum = (float2)(0.0f, 0.0f);

    GRID_STRIDE_INIT();
    GRID_STRIDE_IJ(
        dV_accum += velocity_updates[IDX];
    );

    float2 wg_dV = FLOAT2_WG_REDUCE_ADD(dV_accum);
    if (lane == 0) {
        velocity_intermediate[i] = velocity[i] + ((fast_length(wg_dV) <= DV_MAX) ? wg_dV : 0.0f);
    }
}