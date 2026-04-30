#include "kernel/stride.clh"

__kernel void assign_collision_groups(
               const uint    N,
    __global   const float*  restrict mass,
    __global   const float*  restrict velocity_relative,
    __global   const float*  restrict distance_edge,
    __global         uint*   restrict collision_group
) {
    uint i = get_group_id(0);             
    uint lane = get_local_id(0);          
    uint Lx   = get_local_size(0);        
    uint max_index = i;

    STRIDE(
        if (distance_edge[IDX] <= EPS_DIST)
            max_index = max(max_index, j);
    );

    uint wg_max_index = work_group_reduce_max(max_index);
    if (lane == 0) {
        collision_group[i] = wg_max_index;
    }
}

__kernel void reduce_collision_groups(
               const uint    N,
    __global   const float*  restrict mass,
    __global         uint*   restrict collision_group,
    __global         uint*   restrict has_updates
) {
    uint i = get_global_id(0);

    // get own collision_group ID
    // if it's itself, then we're done!
    if (i >= N || mass[i] == 0 || i == collision_group[i]) return;

    uint workgroup_id = get_group_id(0);
    uint lane = get_local_id(0);
    if (lane == 0) {
        has_updates[workgroup_id] = false;
    }

    // Walk up however many we can
    // We want the largest group ID for the simultaneous collisions
    uint j = collision_group[i];
    bool updated = false;
    while (j != collision_group[j] && collision_group[j] > j) {
        j = collision_group[j];
        updated = true;
    }
    collision_group[i] = j;

    bool wg_has_updated = work_group_any(updated);
    if (lane == 0) {
        has_updates[workgroup_id] = wg_has_updated;
    }
}

__kernel void compute_merging_collision(
               const uint    N,
    __global   const uint*   restrict collision_group,
    __global   const float2* restrict position,
    __global   const float2* restrict velocity,
    __global   const float*  restrict mass,
    __global         float2* restrict position_out,
    __global         float2* restrict velocity_out,
    __global         float*  restrict mass_out
) {
    // one group_id per collision-group
    // most collions-groups will be of size 1 (group of just itself)
    uint i = get_group_id(0);
    if (i >= N || mass[i] == 0) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    if (collision_group[i] != i) {
        // this item is part of a different collision_group, so we can safely set its mass to 0 early
        if (lane == 0) mass_out[i] = 0;
        return;
    }
    
    float2 total_mv = 0;
    float2 total_mr = 0;
    float total_mass = 0;
    uint max_index = i;
    bool is_merging = false;

    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        if (j == i || collision_group[j] != i || mass[j] == 0) continue;
        total_mass += mass[j];
        total_mv += (mass[j] * velocity[j]);
        total_mr += (mass[j] * position[j]);
        is_merging = true;
    }
  
    bool wg_is_merging = work_group_any(is_merging);
    float wg_mass = work_group_reduce_add(total_mass);
    float2 wg_mv = FLOAT2_WG_REDUCE_ADD(total_mv);
    float2 wg_mr = FLOAT2_WG_REDUCE_ADD(total_mr);

    if (lane == 0) {
        mass_out[i] = mass[i];
        velocity_out[i] = velocity[i];
        position_out[i] = position[i];

        if (wg_is_merging) {
            mass_out[i] = mass[i] + wg_mass;

            // Center-of-mass velocity
            velocity_out[i] = (wg_mv + (mass[i] * velocity[i])) / mass_out[i];
            
            // Center-of-mass position
            position_out[i] = (wg_mr + (mass[i] * position[i])) / mass_out[i];
        }
    }
}


__kernel void apply_merge(
    __global   const float*  restrict mass_in,
    __global   const float2* restrict velocity_in,
    __global   const float2* restrict position_in,
    __global         float*  restrict mass_out,
    __global         float2* restrict velocity_out,
    __global         float2* restrict position_out,
    __global         float*  restrict radius
) {
    uint i = get_global_id(0);
    mass_out[i] = mass_in[i];
    velocity_out[i] = velocity_in[i];
    position_out[i] = position_in[i];
    radius[i] = cbrt(mass_out[i] / 3.14159f);
}