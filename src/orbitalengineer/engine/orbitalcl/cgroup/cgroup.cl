#include "kernel/stride.clh"
#include "flags.clh"



__kernel void find_contacting_bodies(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float*  restrict radius,
    __global       float*  restrict edge_dist,
    __global       uint*   restrict num_contacts_per_lane, // (N * lane_count)
    __global       uint*   restrict contacts               // (N * N)
) {
    uint i = get_group_id(0);
    uint lane = get_local_id(0);
    uint Lx = get_local_size(0);

    if (i >= N) return;
    uint row_start = i * N;

    uint lane_width = (uint)ceil(N / (Lx * 1.0));
    uint lane_offset = lane * lane_width;

    uint num_contacts = 0;    
    for (uint j = lane; j < N; j += Lx) {
        if (j == i) continue;
     
        float edge_dist = fast_length(position[j] - position[i]) - radius[i] - radius[j];
        if ((flags[j]&REMOVED) || edge_dist >= max(radius[i], radius[j]) * 0.25) {
            continue;
        }

        contacts[row_start + lane_offset + num_contacts] = j;
        num_contacts++;
    }
    num_contacts_per_lane[(Lx * i) + lane] = num_contacts;
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


// global work siz e = (len(ids), )
// local work size  = (max(num_contacts), )
__kernel void cgroup_assign(
             const uint  N,
    __global const uint* restrict ids_reduced,      // (num_ids, )
    __global const uint* restrict num_contacts,     // (num_ids, )
    __global       uint* restrict contacts_reduced, // (N * N)
    __global const uint* restrict cgroup_src,       
    __global       uint* restrict cgroup_dest,      
    __global       uint* restrict has_updates       
) {
    uint gid = get_global_id(0);
    uint i = ids_reduced[gid];

    // each lane gets a different contacting body

    for (uint n=0; n < num_contacts[i]; n++) {
        uint j = contacts_reduced[(N * i) + n];
        
        while (cgroup_src[j] < j) {
            j = cgroup_src[j];
            contacts_reduced[(N * i) + n] = j;
            has_updates[i] = true;
        }

        if (j < cgroup_src[i]) {
            cgroup_dest[i] = j;
            has_updates[i] = true;
        }
    }

    //bool wg_has_updated = work_group_any(updated);
    //uint wg_min_index = work_group_reduce_min(min_index);
    //if (get_local_id(0) == 0 && wg_has_updated) {
    //if (updated) {
    //    cgroup[i] = min_index;
    //    has_updates[i] = updated;
        //printf("Recording cgroup[%u] = %u, updated=%u", i, cgroup[i], updated);
    //}

    //}
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


// __kernel void compute_edge_distance(
//                const uint    N,
//     __global   const float2* restrict position,
//     __global   const float*  restrict radius,
//     __global         float*  restrict edge_dist
// ) {
//     GRID_STRIDE_INIT();
//     GRID_STRIDE_IJ(
//         edge_dist[IDX] = fast_length(position[j] - position[i]) - radius[i] - radius[j];
//     );
// }

// __kernel void collision_group_assign(
//                const uint    N,
//     __global   const uint*   restrict flags,
//     __global   const float*  restrict radius,
//     __global   const float*  restrict edge_dist,
//     __global   const uint*   restrict cgroup,
//     __global         uint*   restrict new_cgroup,
//     __global         uint*   restrict has_updates
// ) {
//     GRID_STRIDE_INIT();
//     if (lane == 0) new_cgroup[i] = cgroup[i];
//     if ((flags[i]&REMOVED)) return;


//     uint min_index = cgroup[i];
//     bool updated = false;

//     GRID_STRIDE_IJ(
//         if ((flags[j]&REMOVED) || edge_dist[IDX] >= max(radius[i], radius[j]) * 0.25) {
//             continue;
//         } else if (cgroup[j] < cgroup[i]) {
//             min_index = cgroup[j];
//             updated = true;
//         }
//     );

//     while (cgroup[min_index] < min_index) {
//         min_index = cgroup[min_index];
//         updated = true;
//     }

//     bool wg_has_updated = work_group_any(updated);
//     uint wg_min_index = work_group_reduce_min(min_index);
//     if (lane == 0) {
//         new_cgroup[i] = wg_min_index;
//         has_updates[i] = wg_has_updated;
//     }
// }
 