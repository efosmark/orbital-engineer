#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/debug.clh"


__kernel void find_contacting_bodies(
             const uint  N_bodies_alloc,
             const uint  N_bodies_valid,
    __global const uint* restrict flags,
    __global const bool* restrict is_touching,
    __global uint* num_contacts,
    __global uint* contacts,
    __global uint* num_contacts,
    __global uint* contacts,
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    __local atomic_uint local_count;
    __local uint contacting_i[MAX_NUM_CONTACTS_PER_BODY];

    if (get_local_id(0) == 0)
        atomic_init(&local_count, 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    GRID_STRIDE_IJ(
        bool in_contact = (!(flags[j]&REMOVED) && is_touching[IDX]);
        if (in_contact) {
            uint offset = atomic_fetch_add(&local_count, 1);
            if (in_contact && offset < MAX_NUM_CONTACTS_PER_BODY)
                contacting_i[offset] = j;
        }
    );

    barrier(CLK_LOCAL_MEM_FENCE);

    uint offset = atomic_fetch_add(&local_count, 0);
    for (uint jx=lane; jx<offset && jx < MAX_NUM_CONTACTS_PER_BODY; jx += Lx) {
        contacts[(i * N) + jx] = contacting_i[jx];
    }
    num_contacts[i] = offset;
}
