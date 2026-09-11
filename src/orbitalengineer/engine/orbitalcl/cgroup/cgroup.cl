#include "kernel/stride.clh"
#include "flags.clh"


__kernel void cgroup_assign(
             const uint  N,
    __global const uint* restrict ids,      // (num_ids, )
    __global const uint* restrict n_direct_contacts,     // (num_ids, )
    __global const uint* restrict direct_contacts, // (N * N)
    __global const uint* restrict cgroup_src,       
    __global       uint* restrict cgroup_dest,      
    __global       uint* restrict has_updates       
) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    uint Lx = get_local_size(0);

    uint i = ids[gid];
    bool updated = false;
    uint min_contact = i;

    // Ensure this lane is in-bounds
    if (lid < n_direct_contacts[i]) {

        // Our current min ID based on our collision neighbors
        min_contact = cgroup_src[i];

        // Get the individual collision
        uint j = direct_contacts[(N * i) + lid];

        // Look up its current min ID based on _its_ collision neighbors
        uint j_min_contact = cgroup_src[j];

        // Check whether it is smaller than our current min ID
        while (j_min_contact < min_contact) {
            min_contact = j_min_contact;
            updated = true;
            j_min_contact = cgroup_src[j_min_contact];
        }
    }

    bool wg_has_updated = work_group_any(updated);
    uint wg_min_index = work_group_reduce_min(min_contact);
    if (lid == 0) {
        cgroup_dest[i] = wg_min_index;
        has_updates[i] = wg_has_updated;
    }
}


__kernel void organize_cgroup_state_vectors(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const uint*   restrict cgroup,
    __global       float2* restrict p_com_by_id,
    __global       float2* restrict v_com_by_id,
    __global       float*  restrict m_com_by_id
) {
    uint i = get_global_id(0);
    bool i_enabled = (flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0;
    if (!i_enabled) return;

    uint idx = (cgroup[i] * N) + i;
    
    p_com_by_id[idx] = position[i];
    v_com_by_id[idx] = velocity[i];
    m_com_by_id[idx] = mass[i];
}



// __kernel void combine_cgroup_state_vectors(
//              const uint    N,
//     __global const uint*   restrict cgroup,
//     __global const float2* restrict p_com_by_id,
//     __global const float2* restrict v_com_by_id,
//     __global const float*  restrict m_com_by_id,
//     __global       float2* restrict p_com,
//     __global       float2* restrict v_com,
//     __global       float*  restrict m_com
// ) {
//     uint i = get_group_id(0);
//     bool i_enabled = (flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0;
//     if (!i_enabled) return;



//     uint idx = (cgroup[i] * N) + i;
    
//     p_com[i] = p_com_by_id[i];
//     v_com[i] = v_com_by_id[i];
//     m_com[i] = m_com_by_id[i];
// }
