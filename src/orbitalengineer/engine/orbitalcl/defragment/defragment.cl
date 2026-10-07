#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/debug.clh"
#include "kernel/ledger.clh"


// work_group_global_size = (N / Lx, )
// work_group_local_size  = (Lx,     )
__kernel void defrag_orbital_vectors(
             const uint    N_curr,
    __global atomic_uint*  global_count,

    __global const uint*   id_to_index,
    __global const uint*   body_id,
    __global const uint*   flags,
    __global const float2* position,
    __global const float2* velocity,
    __global const float*  mass,
    __global const float*  radius,

    __global       uint*   id_to_index_out,
    __global       uint*   body_id_out,
    __global       uint*   flags_out,
    __global       float2* position_out,
    __global       float2* velocity_out,
    __global       float*  mass_out,
    __global       float*  radius_out
) {
    uint i = get_global_id(0);
    uint lane = get_local_id(0);

    uint removed_count = atomic_load(global_count);

    // if (get_global_id(0) == 0 && get_local_id(0) == 0) {
    //    atomic_init(N_removed, 0);
    // }

    // uint num_removed = 0;
    // for (uint i=get_local_id(0); i < N_curr; i += get_local_size(0)) {

    // }

    id_to_index_out[i] = id_to_index[i];
    body_id_out[i] = body_id[i];
    flags_out[i] = flags[i];
    position_out[i] = position[i];
    velocity_out[i] = velocity[i];
    mass_out[i] = mass[i];
    radius_out[i] = radius[i];

    uint should_emit = (flags[i]&REMOVED) && (i < (N_curr - removed_count));

    uint emit = should_emit ? 1 : 0;

    // Where am I among the emitting lanes?
    uint offset = work_group_scan_exclusive_add(emit);

    // How many events did this workgroup produce?
    uint count = work_group_reduce_add(emit);

    // One atomic allocation for the entire workgroup.
    uint base = 0;
    uint current_global_count;
    if (get_local_id(0) == 0) {
        base = atomic_fetch_add(global_count, count);
        current_global_count = atomic_load(global_count);
    }
    base = work_group_broadcast(base, 0);
    current_global_count = work_group_broadcast(current_global_count, 0);

    if (i >= N_curr - base) {
        return;
    }

    // if (get_local_id(0) == 0 && count > 0) {
    //     printf("[wg=%u  count=%u  global_count=%u  base=%u]", (uint)get_group_id(0), count, removed_count, base);
    // }

    base = base + count;


    // Every emitting lane now has a unique global destination.
    if (emit && count) {
        uint j = (N_curr - base) + offset;




        if (j <= i || (flags[j]&REMOVED)) {
            return;
        }
        //DEBUG_PRINTF("[g0=%u, l0=%u] Swapping %u -> %u", (uint)get_global_id(0), (uint)get_local_id(0), i, j);

        if ((flags[j]&REMOVED))
           DEBUG_PRINTF("[g0=%u, l0=%u] Swapping %u -> %u FAILED", (uint)get_global_id(0), (uint)get_local_id(0), i, j);


        //Move j values to slot i
        id_to_index_out[j] = i;
        body_id_out[i] = body_id[j];
        flags_out[i] = flags[j];
        position_out[i] = position[j];
        velocity_out[i] = velocity[j];
        mass_out[i] = mass[j];
        radius_out[i] = radius[j];

        // Move i values to slot j
        id_to_index_out[i] = j;
        body_id_out[j] = body_id[i];
        flags_out[j] = flags[i];
        position_out[j] = position[i];
        velocity_out[j] = velocity[i];
        mass_out[j] = mass[i];
        radius_out[j] = radius[i];

        //DEBUG_PRINTF("(%u, %u) flags_out[%u]=%u  flags_out[%u]=%u", i, j, i, flags_out[i], j, flags_out[j]);
    }
}
