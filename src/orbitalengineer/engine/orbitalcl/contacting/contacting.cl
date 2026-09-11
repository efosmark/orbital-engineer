#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/debug.clh"


__kernel void find_contacting_bodies(
             const uint    N,
    __global const uint*   restrict flags,
    __global const bool*   restrict is_touching,
    __global       uint*   restrict num_contacts_by_lane,
    __global       uint*   restrict contacts_by_lane
) {
    GRID_STRIDE_INIT();
    if (flags[i]&REMOVED) return;

    uint lane_width = (uint) ceil(N / (Lx * 1.0));
    uint lane_offset = lane * lane_width;
    uint num_contacts = 0;

    GRID_STRIDE_IJ(
        uint lane_start = row_start + lane_offset;
        bool in_contact = (!(flags[j]&REMOVED) && is_touching[IDX]);
        contacts_by_lane[lane_start + num_contacts] = in_contact ? j : 0;
        num_contacts += in_contact ? 1 : 0;
    );
    num_contacts_by_lane[(Lx * i) + lane] = num_contacts;
}


__kernel void find_contacting_bodies_reduce(
             const uint  N,
             const uint  Lx,
    __global const uint* restrict flags,
    __global const uint* restrict num_contacts_by_lane,
    __global const uint* restrict contacts_by_lane,
    __global       uint* restrict num_contacts,
    __global       uint* restrict contacts_reduced
) {
    uint i = get_global_id(0);
    if (flags[i]&REMOVED) return;

    uint row_start = i * N;
    uint lane_width = (uint) ceil(N / (Lx * 1.0));
    uint n_contact = 0;
    for(uint lane = 0; lane < Lx && n_contact < MAX_NUM_CONTACTS_PER_BODY; lane++) {
        uint lane_offset = lane * lane_width;
        uint lane_start = row_start + lane_offset;

        // `k` rarely goes above 1 when N < 1e3
        __attribute__((opencl_unroll_hint))
        for (uint k = 0; k < num_contacts_by_lane[(Lx * i) + lane] && n_contact < MAX_NUM_CONTACTS_PER_BODY; k++) {
            uint j = contacts_by_lane[lane_start + k];
            contacts_reduced[row_start + n_contact] = j;
            n_contact++;

            DEBUG_PRINTF("(contacting) [%u] => %u", i, j);
        }
    }
    num_contacts[i] = n_contact;
}
