#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/pairwise.clh"


__kernel void find_contacting_bodies_orig(
             const uint    N,
    __global const Pair*   restrict pairs,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict radius,
    __global const float2* restrict time_of_interaction,
    __global       uint*   restrict num_contacts_per_lane, // (N * lane_count)
    __global       uint*   restrict contacts_per_lane      // (N * N)
) {
    GRID_STRIDE_INIT();

    uint lane_width = (uint) (N / (Lx * 1.0));
    uint lane_offset = lane * lane_width;
    uint num_contacts = 0;

    float2 position_i = position[i];
    float radius_i = radius[i];

    GRID_STRIDE_IJ(
        float2 ttc_ij = time_of_interaction[row_start + j];
        float ttc_min = fmin(fabs(ttc_ij.x), fabs(ttc_ij.y));

        float edge_dist = fast_distance(position_i, position[j]) - radius_i - radius[j];

        if ((flags[j]&REMOVED) || edge_dist > fmin(radius_i, radius[j]) * 0.1) {
            continue;
        } 

        contacts_per_lane[row_start + lane_offset + num_contacts] = j;
        num_contacts++;
    );
    num_contacts_per_lane[(Lx * i) + lane] = num_contacts;
}


//#define MAX_NUM_COLLISIONS_PER_LANE 8
//#define MAX_NUM_COLLISIONS 24


__kernel void find_contacting_bodies(
             const uint    N,
             const uint    num_paris,
    __global const Pair*   restrict pair,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict radius,
    __global const float2* restrict time_of_interaction,
    __global       uint*   restrict num_contacts_per_lane, // (N * lane_count)
    __global       uint*   restrict contacts_per_lane      // (N * N)
) {

    uint g0 = get_group_id(0);
    //if (g0 >= num_paris) return;

    uint lane = get_local_id(0);
    uint Lx = get_local_size(0);

    uint lane_width = (uint) (N / (Lx * 1.0));
    uint lane_offset = lane * lane_width;
    
    printf("g0=%u, lane=%u, Lx=%u", g0, lane, Lx);
    for (uint n = lane; n < num_paris; n += Lx) {
        uint i = pair[((g0 * Lx) + n)].i;
        uint j = pair[((g0 * Lx) + n)].j;

        //printf("i=%u, j=%u, n=%u, g0=%u, lane=%u", i, j, n, g0, lane);

        float2 ttc_ij = time_of_interaction[(N * i) + j];
        float ttc_min = fmin(fabs(ttc_ij.x), fabs(ttc_ij.y));

        float edge_dist = fast_distance(position[i], position[j]) - radius[i] - radius[j];
        if ((flags[j]&REMOVED) || edge_dist > fmin(radius[i], radius[j]) * 0.5) continue;

        uint ncontacts_i = num_contacts_per_lane[(Lx * i) + lane];
        uint ncontacts_j = num_contacts_per_lane[(Lx * j) + lane];

        contacts_per_lane[(N * i) + lane_offset + ncontacts_i] = j;
        contacts_per_lane[(N * j) + lane_offset + ncontacts_j] = i;

        num_contacts_per_lane[(Lx * i) + lane]++;
        num_contacts_per_lane[(Lx * j) + lane]++;
    }
}


__kernel void find_contacting_bodies_reduce(
             const uint  N,
             const uint  Lx,
    __global const uint* restrict num_contacts_per_lane, // (N * Lx,  )
    __global const uint* restrict contacts,              // (N * N,   )
    __global       uint* restrict num_contacts,          // (N,       )
    __global       uint* restrict contacts_reduced       // (N * N,   )
) {
    uint i = get_global_id(0);
    if (i >= N) return;

    uint row_start = i * N;
    uint lane_width = (uint) ceil(N / (Lx * 1.0));
    uint n_contact = 0;
    for(uint lane = 0; lane < Lx; lane++) {
        uint lane_offset = lane * lane_width;
        for (uint k = 0; k < num_contacts_per_lane[(Lx * i) + lane]; k++) {
            contacts_reduced[row_start + n_contact] = contacts[row_start + lane_offset + k];
            n_contact++;
        }
    }
    num_contacts[i] = n_contact;
}


__kernel void cgroup_assign(
             const uint  N,
    __global const uint* restrict ids_reduced,      // (num_ids, )
    __global const uint* restrict num_contacts,     // (num_ids, )
    __global const uint* restrict contacts_reduced, // (N * N)
    __global const uint* restrict cgroup_src,       
    __global       uint* restrict cgroup_dest,      
    __global       uint* restrict has_updates       
) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    uint Lx = get_local_size(0);

    uint i = ids_reduced[gid];
    bool updated = false;
    uint min_contact = i;

    // Ensure this lane is in-bounds
    if (lid < num_contacts[i]) {

        // Our current min ID based on our collision neighbors
        min_contact = cgroup_src[i];

        // Get the individual collision
        uint j = contacts_reduced[(N * i) + lid];

        // Look up its current min ID based on _its_ collision neighbors
        uint j_min_contact = cgroup_src[j];

        // Check whether it is smaller than our current min ID
        if (j_min_contact < min_contact) {
            min_contact = j_min_contact;
            updated = true;
        }
    }

    bool wg_has_updated = work_group_any(updated);
    uint wg_min_index = work_group_reduce_min(min_contact);
    if (lid == 0) {
        cgroup_dest[i] = wg_min_index;
        has_updates[i] = wg_has_updated;
    }
}
