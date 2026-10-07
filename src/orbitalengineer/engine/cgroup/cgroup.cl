#include "kernel/stride.clh"
#include "flags.clh"


__kernel void cgroup_assign(
             const uint  N_bodies_alloc,
    __global const uint* ids,
    __global const uint* n_direct_contacts,
    __global const uint* direct_contacts,
    __global const uint* cgroup_src,       
    __global       uint* cgroup_dest,
    __global atomic_uint* n_updates
) {
    uint gid = get_group_id(0);  // One group per node
    uint lid = get_local_id(0);  // One lane per edge
    uint Lx = get_local_size(0);

    if (get_local_id(0) == 0 && get_global_id(0) == 0) {
        atomic_init(n_updates, 0);
    }

    uint i = ids[gid];
    bool updated = false;
    uint min_contact = i;

    // Ensure this lane is in-bounds
    if (lid < n_direct_contacts[i]) {

        // Our current min ID based on our collision neighbors
        min_contact = cgroup_src[i];

        // Get the individual collision
        uint j = direct_contacts[(N_bodies_alloc * i) + lid];

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
        atomic_fetch_add(n_updates, (uint)wg_has_updated);
    }
}
