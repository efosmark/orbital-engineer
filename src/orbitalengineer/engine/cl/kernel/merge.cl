#include "kernel/stride.clh"
#include "flags.clh"

#define SUPPORTS_MERGE(i) ((flags[i]&MERGE_AS_PRIMARY) || (flags[i]&MERGE_AS_SECONDARY))

__kernel void compute_merging_collision(
             const uint    N,
    __global       uint*   restrict flags,
    __global const uint*   restrict collision_group,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global       float2* restrict position_out,
    __global       float2* restrict velocity_out,
    __global       float*  restrict mass_out
) {
    GRID_STRIDE_INIT();
    if ((flags[i]&REMOVED)) {
        return;

    } else if (collision_group[i] != i && (flags[i]&MERGE_AS_SECONDARY) && (flags[collision_group[i]]&MERGE_AS_PRIMARY)) {
        if (lane == 0) {
            // this item is part of a different collision_group, so we can safely set its mass to 0 early
            mass_out[i] = 0.0f;
            flags[i] |= REMOVED;
        }
        return;

    } else if ((flags[i]&MERGE_AS_PRIMARY) == 0) {
        if (lane == 0) {
            mass_out[i] = mass[i];
            velocity_out[i] = velocity[i];
            position_out[i] = position[i];
        }
        return;
    }

    float total_mass = 0;
    float2 total_mv = 0;
    float2 total_mr = 0;
    bool is_merging = false;

    GRID_STRIDE_IJ(
        if (collision_group[j] != i || (flags[j]&REMOVED) || !(flags[j]&MERGE_AS_SECONDARY)) continue;
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
        mass_out[i] = mass[i];
        velocity_out[i] = velocity[i];
        position_out[i] = position[i];

        if (wg_is_merging) {
            mass_out[i] += (flags[i]&FIXED_MASS) ? 0 : wg_mass;

            // Center-of-mass velocity
            float2 velocity_center_of_mass = (wg_mv + (mass[i] * velocity[i])) / mass_out[i];
            velocity_out[i] = (flags[i]&FIXED_VELOCITY) ? velocity[i] : velocity_center_of_mass;
            
            // Center-of-mass position
            float2 position_center_of_mass = (wg_mr + (mass[i] * position[i])) / mass_out[i];
            position_out[i] = (flags[i]&FIXED_POSITION) ? position[i] : position_center_of_mass;
        }
    }
}


__kernel void apply_merge(
    __global const uint*   restrict flags,
    __global const float*  restrict mass_in,
    __global const float2* restrict velocity_in,
    __global const float2* restrict position_in,
    __global       float*  restrict mass_out,
    __global       float2* restrict velocity_out,
    __global       float2* restrict position_out,
    __global       float*  restrict radius_out
) {
    uint i = get_global_id(0);
    mass_out[i] = mass_in[i];
    velocity_out[i] = velocity_in[i];
    position_out[i] = position_in[i];

    radius_out[i] = (flags[i]&FIXED_RADIUS) ? radius_out[i] : cbrt(mass_out[i] / 3.14159f);
}