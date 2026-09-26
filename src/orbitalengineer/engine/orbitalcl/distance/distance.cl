#include "kernel/stride.clh"
#include "flags.clh"

__kernel void edge_distance(
             const uint    N,
    __global const uint*   flags,
    __global const float2* position,
    __global const float2* velocity,
    __global const float*  radius,
    __global       float*  edge_distance,
    __global       bool*   is_touching,
    __global       bool*   is_nearby,
    __global       atomic_uint* n_all_colliding_ids,
    __global       uint*   all_colliding_ids,
    __global       uint*   num_contacts,
    __global       uint*   contacts,
    __global       uint*   num_nearby,
    __global       uint*   nearby
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    __local atomic_uint local_num_contacts;
    __local atomic_uint local_num_nearby;
    __local uint contacting_i[MAX_NUM_CONTACTS_PER_BODY];
    __local uint nearby_i[MAX_NUM_CONTACTS_PER_BODY];

    if (get_local_id(0) == 0) {
        atomic_init(&local_num_contacts, 0);
        atomic_init(&local_num_nearby, 0);
        if (get_global_id(0) == 0)
            atomic_init(n_all_colliding_ids, 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    GRID_STRIDE_IJ(
        if (flags[j]&REMOVED) continue;
        float R = radius[j] + radius[i];
        float2 dV = velocity[j] - velocity[i];
        float2 dP = position[j] - position[i];
        float distance_edge_to_edge = fast_length(dP) - R;

        bool in_contact = distance_edge_to_edge <= EPS_DIST;
        if (in_contact) {
            uint offset = atomic_fetch_add(&local_num_contacts, 1);
            if (offset < MAX_NUM_CONTACTS_PER_BODY)
                contacting_i[offset] = j;
        }

        bool is_near = distance_edge_to_edge <= R * 0.1;
        if (is_near) {
            uint offset = atomic_fetch_add(&local_num_nearby, 1);
            if (offset < MAX_NUM_CONTACTS_PER_BODY)
                nearby_i[offset] = j;
        }

        is_touching[IDX] = in_contact;
        is_nearby[IDX] = is_near;
        edge_distance[IDX] = distance_edge_to_edge;
    );

    barrier(CLK_LOCAL_MEM_FENCE);

    uint offset = atomic_fetch_add(&local_num_contacts, 0);
    for (uint jx=lane; jx < offset && jx < MAX_NUM_CONTACTS_PER_BODY; jx += Lx) {
        uint index = (i * N) + jx;
        contacts[index] = contacting_i[jx];
    }
    num_contacts[i] = offset;

    offset = atomic_fetch_add(&local_num_nearby, 0);
    for (uint jx=lane; jx < offset && jx < MAX_NUM_CONTACTS_PER_BODY; jx += Lx) {
        uint index = (i * N) + jx;
        nearby[index] = nearby_i[jx];
    }
    num_nearby[i] = offset;

    if (lane == 0 && offset > 0) {
        offset = atomic_fetch_add(n_all_colliding_ids, 1);
        all_colliding_ids[offset] = i;
    }
}