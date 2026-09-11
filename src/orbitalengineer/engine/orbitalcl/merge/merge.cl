#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/debug.clh"

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


// __kernel void compute_merging_collision(
//              const uint    N,
//     __global const uint*   restrict flags,
//     __global const uint*   restrict merge_group,
//     __global const float2* restrict position,
//     __global const float2* restrict velocity,
//     __global const float*  restrict mass,
//     __global const float*  restrict radius,
//     __global       uint*   restrict flags_out,
//     __global       float2* restrict position_out,
//     __global       float2* restrict velocity_out,
//     __global       float*  restrict mass_out,
//     __global       float*  restrict radius_out
// ) {
//     GRID_STRIDE_INIT();

//     if (flags[i]&REMOVED) {
//         return;
//     }

//     if ((merge_group[i] != i) && (flags[i]&MERGE_AS_SECONDARY) && (flags[merge_group[i]]&MERGE_AS_PRIMARY)) {
//         if (lane == 0) {
//             mass_out[i] = (flags[i]&FIXED_MASS) ? mass[i] : 0;
//             flags_out[i] = flags[i]|REMOVED;
//         }
//         return;
//     }

//     if ((flags[i]&MERGE_AS_PRIMARY) == 0) {
//         if (lane == 0) {
//             flags_out[i] = flags[i];
//             mass_out[i] = mass[i];
//             velocity_out[i] = velocity[i];
//             position_out[i] = position[i];
//             radius_out[i] = radius[i];
//         }
//         return;
//     }

//     float total_mass = 0;
//     float2 total_mv = 0;
//     float2 total_mr = 0;
//     bool is_merging = false;

//     GRID_STRIDE_IJ(
//         if (merge_group[j] != i || (flags[j]&REMOVED) || !(flags[i]&MERGE_AS_SECONDARY)) continue;
//         total_mass += mass[j];
//         total_mv += (mass[j] * velocity[j]);
//         total_mr += (mass[j] * position[j]);
//         is_merging = true;
//     );
  
//     float wg_mass = work_group_reduce_add(total_mass);
//     float2 wg_mv = FLOAT2_WG_REDUCE_ADD(total_mv);
//     float2  wg_mr = FLOAT2_WG_REDUCE_ADD(total_mr);
//     bool wg_is_merging = work_group_any(is_merging);

//     if (lane == 0) {
//         flags_out[i] = flags[i];
//         mass_out[i] = mass[i];
//         velocity_out[i] = velocity[i];
//         position_out[i] = position[i];
//         radius_out[i] = radius[i];

//         if (wg_is_merging) {
//             mass_out[i] += (flags[i]&FIXED_MASS) ? 0 : wg_mass;

//             // Center-of-mass velocity
//             float2 velocity_center_of_mass = (wg_mv + (mass[i] * velocity[i])) / mass_out[i];
//             velocity_out[i] = (flags[i]&FIXED_VELOCITY) ? velocity[i] : velocity_center_of_mass;
            
//             // Center-of-mass position
//             float2 position_center_of_mass = (wg_mr + (mass[i] * position[i])) / mass_out[i];
//             position_out[i] = (flags[i]&FIXED_POSITION) ? position[i] : position_center_of_mass;

//             // New radius based on updated mass
//             radius_out[i] = (flags[i]&FIXED_RADIUS) ? radius_out[i] : cbrt(mass_out[i] / 3.14159f);
//         }
//     }
// }

__kernel void compute_merging_collision_direct(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const uint*   restrict num_contacts,
    __global const uint*   restrict direct_contacts,
    __global       uint*   restrict flags_out,
    __global       float2* restrict position_out,
    __global       float2* restrict velocity_out,
    __global       float*  restrict mass_out,
    __global       float*  restrict radius_out
) {
    GRID_STRIDE_INIT();

    if (lane == 0) {
        flags_out[i] = flags[i];
        mass_out[i] = mass[i];
        velocity_out[i] = velocity[i];
        position_out[i] = position[i];
        radius_out[i] = radius[i];
    }

    if ((flags[i]&REMOVED) || (flags[i]&MERGE_AS_PRIMARY) == 0) return;

    uint j = (lane < num_contacts[i]) ? direct_contacts[(N * i) + lane] : i; 
    bool is_merging = (j != i) && !(flags[j]&REMOVED) && (flags[i]&MERGE_AS_SECONDARY);
    
    float total_mass = is_merging ? mass[j] : 0;
    float2 total_mv = is_merging ? (mass[j] * velocity[j]) : 0;
    float2 total_mr = is_merging ? (mass[j] * position[j]) : 0;
    
    float wg_mass = work_group_reduce_add(total_mass);
    float2 wg_mv = FLOAT2_WG_REDUCE_ADD(total_mv);
    float2  wg_mr = FLOAT2_WG_REDUCE_ADD(total_mr);
    bool wg_is_merging = work_group_any(is_merging);
    uint wg_leader = work_group_reduce_min(wg_is_merging ? j : INT_MAX);

    if (wg_is_merging && i == wg_leader && lane == 0) {
        DEBUG_PRINTF("(coalesce) [%u]", wg_leader);
        mass_out[i] += (flags[i]&FIXED_MASS) ? 0 : wg_mass;

        // Center-of-mass velocity
        float2 velocity_center_of_mass = (wg_mv + (mass[i] * velocity[i])) / mass_out[i];
        velocity_out[i] = (flags[i]&FIXED_VELOCITY) ? velocity[i] : velocity_center_of_mass;
        
        // Center-of-mass position
        float2 position_center_of_mass = (wg_mr + (mass[i] * position[i])) / mass_out[i];
        position_out[i] = (flags[i]&FIXED_POSITION) ? position[i] : position_center_of_mass;

        // New radius based on updated mass
        radius_out[i] = (flags[i]&FIXED_RADIUS) ? radius_out[i] : cbrt(mass_out[i] / 3.14159f);

    } else if (wg_is_merging && lane == 0) {
        DEBUG_PRINTF("(coalesce) [%u] <-- %u", i, wg_leader);
        mass_out[i] = (flags[i]&FIXED_MASS) ? mass[i] : 0;
        flags_out[i] = flags[i]|REMOVED;
    }
}