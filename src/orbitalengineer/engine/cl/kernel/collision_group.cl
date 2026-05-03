#include "kernel/stride.clh"


__kernel void assign_collision_groups(
               const uint    N,
               const float   dt,
    __global   const float*  restrict mass,
    __global   const float*  restrict velocity_relative,
    __global   const float*  restrict distance_edge,
    __global         uint*   restrict collision_group
) {
    uint i = get_group_id(0);
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);
    uint max_index = i;

    GRID_STRIDE_IJ(
        if (mass[j] == 0) continue;

        if (distance_edge[IDX] <= EPS_DIST)
            max_index = min(max_index, j);
    );

    uint wg_max_index = work_group_reduce_min(max_index);
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

    // Get own collision_group ID. If it's itself, then we're done!
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
