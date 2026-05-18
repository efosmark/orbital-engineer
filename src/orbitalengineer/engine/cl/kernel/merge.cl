#include "kernel/stride.clh"
#include "flags.clh"


/**
 * Each node compares itself to every other node. If it finds that it is overlapping,
 * choose whichever has the lowest index as the group ID.
 * 
 * Flags used:
 *  - REMOVED            -- Skip the node from being processed.
 *  - MERGE_AS_PRIMARY   -- Ensure the node can be the group leader.
 *  - MERGE_AS_SECONDARY -- Ensure the node can be absorbed by a merge operation.
 */
__kernel void collide_merge_group_assign(
               const uint    N,
               const float   dt,
    __global   const uint*   restrict flags,
    __global   const float*  restrict mass,
    __global   const float*  restrict velocity_relative,
    __global   const float*  restrict edge_distance,
    __global         uint*   restrict merge_group
) {
    GRID_STRIDE_INIT();
    if ((flags[i]&REMOVED)) return;
    
    uint min_index = i;

    GRID_STRIDE_IJ(
        if ((flags[j]&REMOVED) || edge_distance[IDX] > EPS_DIST) {
            continue;
        } else if ((flags[j]&MERGE_AS_PRIMARY) && (flags[i]&MERGE_AS_SECONDARY) && j < i) {
            min_index = j;
        }
    );

    uint wg_min_index = work_group_reduce_min(min_index);
    if (lane == 0) {
        merge_group[i] = wg_min_index;
    }
}


/**
 * Look at every node and scan the other members of the group to find the node with the lowest index.
 * This is to consolidate situations where 'A' is touching 'B', 'B' is touching 'C', but 'A' is not
 * touching 'C'.
 */
__kernel void collide_merge_group_reduce(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float*  restrict mass,
    __global       uint*   restrict merge_group,
    __global       uint*   restrict has_updates
) {
    uint i = get_global_id(0);

    // Get own merge_group ID. If it's itself, then we're done!
    if (i >= N || (flags[i]&REMOVED) || i == merge_group[i])
        return;

    uint workgroup_id = get_group_id(0);
    uint lane = get_local_id(0);
    if (lane == 0) has_updates[workgroup_id] = false;

    // Walk up however many we can
    // We want the smallest group ID for the simultaneous collisions
    uint j = merge_group[i];
    bool updated = false;
    while (j != merge_group[j] && merge_group[j] < j && (flags[merge_group[j]]&MERGE_AS_PRIMARY) && (flags[j]&MERGE_AS_SECONDARY) ) {
        j = merge_group[j];
        updated = true;
    }
    
    merge_group[i] = j;
    bool wg_has_updated = work_group_any(updated);
    if (lane == 0) has_updates[workgroup_id] = wg_has_updated;
}


/**
 * Find the center-of-mass for each merge_group and apply them to the group leader.
 * All subordinate nodes gain the 'REMOVED' flag and have their mass set to 0.
 *
 * Flags used:
 *  - REMOVED            -- Skip the node from being processed.
 *  - FIXED_MASS         -- Prevent the node's mass from being changed.
 *  - FIXED_VELOCITY     -- Prevent the node's velocity from being changed.
 *  - FIXED_POSITION     -- Prevent the node's position from being changed.
 */
__kernel void compute_merging_collision(
             const uint    N,
    __global const uint*   restrict flags,
    __global const uint*   restrict merge_group,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       uint*   restrict flags_out,
    __global       float2* restrict position_out,
    __global       float2* restrict velocity_out,
    __global       float*  restrict mass_out,
    __global       float*  restrict radius_out
) {
    GRID_STRIDE_INIT();
    if ((flags[i]&REMOVED) || merge_group[i] != i) {
        if (lane == 0) {
            mass_out[i] = (flags[i]&FIXED_MASS) ? mass[i] : 0;
            flags_out[i] = flags[i]|REMOVED;
        }
        return;
    }

    float total_mass = 0;
    float2 total_mv = 0;
    float2 total_mr = 0;
    bool is_merging = false;

    GRID_STRIDE_IJ(
        if (merge_group[j] != i || (flags[j]&REMOVED)) continue;
        total_mass += mass[j];
        total_mv += (mass[j] * velocity[j]);
        total_mr += (mass[j] * position[j]);
        is_merging = true;
    );
  
    float wg_mass = work_group_reduce_add(total_mass);
    float2 wg_mv = FLOAT2_WG_REDUCE_ADD(total_mv);
    float2  wg_mr = FLOAT2_WG_REDUCE_ADD(total_mr);
    bool wg_is_merging = work_group_any(is_merging);

    if (lane == 0) {
        flags_out[i] = flags[i];
        mass_out[i] = mass[i];
        velocity_out[i] = velocity[i];
        position_out[i] = position[i];
        radius_out[i] = radius[i];

        if (wg_is_merging) {
            mass_out[i] += (flags[i]&FIXED_MASS) ? 0 : wg_mass;

            // Center-of-mass velocity
            float2 velocity_center_of_mass = (wg_mv + (mass[i] * velocity[i])) / mass_out[i];
            velocity_out[i] = (flags[i]&FIXED_VELOCITY) ? velocity[i] : velocity_center_of_mass;
            
            // Center-of-mass position
            float2 position_center_of_mass = (wg_mr + (mass[i] * position[i])) / mass_out[i];
            position_out[i] = (flags[i]&FIXED_POSITION) ? position[i] : position_center_of_mass;

            // New radius based on updated mass
            radius_out[i] = (flags[i]&FIXED_RADIUS) ? radius_out[i] : cbrt(mass_out[i] / 3.14159f);
        }
    }
}

/**
 * Update all of the node vectors from their intermediate values.
 * The radius is dynamically computed from `cbrt(mass / pi)`.
 *
 * Flags used:
 *  - FIXED_RADIUS       -- Prevent the radius from changing.
 */
__kernel void apply_merge(
    __global const uint*   restrict flags,
    __global const float*  restrict mass_in,
    __global const float2* restrict velocity_in,
    __global const float2* restrict position_in,
    __global       uint*   restrict flags_out,
    __global       float*  restrict mass_out,
    __global       float2* restrict velocity_out,
    __global       float2* restrict position_out,
    __global       float*  restrict radius_out
) {
    uint i = get_global_id(0);
    flags_out[i] = flags[i];
    mass_out[i] = mass_in[i];
    velocity_out[i] = velocity_in[i];
    position_out[i] = position_in[i];
    radius_out[i] = (flags[i]&FIXED_RADIUS) ? radius_out[i] : cbrt(mass_out[i] / 3.14159f);
}