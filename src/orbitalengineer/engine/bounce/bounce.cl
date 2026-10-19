#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/complex.clh" // needed for creal, cmul, cconj
#include "kernel/ledger.clh"
#include "dtype.clh"

inline REAL relative_speed_along_normal(
    const uint i,
    const uint j,
    const COMPLEX r_norm,
    const COMPLEX* velocity
) {
    COMPLEX vr = velocity[i] - velocity[j]; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}

__kernel void compute_bouncing_collision_single(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
    __global const uint*   restrict flags,
    __global const COMPLEX* restrict position,
    __global const COMPLEX* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const COMPLEX* restrict time_until_contact,
    __global       COMPLEX* restrict velocity_intermediate,
    __global       atomic_uint* global_count,
    __global LedgerEntry*  restrict ledger
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0) {
        if (lane == 0) {
            velocity_intermediate[i] = velocity[i];
        }
        return;
    }

    float scalar_impulse_magnitude_max = 0.0f;
    uint j_max = i;
    COMPLEX r_norm_max = (COMPLEX)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    GRID_STRIDE_IJ(

        // We only care about bodies that support bouncing
        if ((flags[j]&BOUNCE_AS_SECONDARY) == 0 || (flags[j]&REMOVED)) continue;

        // Limit to only contacting bodies
        COMPLEX toi = time_until_contact[IDX];
        float t1 = toi.x;
        float t2 = toi.y;
        float curr_min = (t1 >= 0.0f) ? t1 : ((t2 >= 0.0f) ? t2 : EPS_TIME);
        if (curr_min > EPS_TIME) continue;

        COMPLEX dP = position[j] - position[i];
        float edge_dist = length(dP) - radius[i] - radius[j];
        if (edge_dist > EPS_DIST) continue;

        COMPLEX r_norm = normalize(dP);

        float v_rel = relative_speed_along_normal(i, j, -r_norm, velocity);
        if (v_rel >= 0) continue;

        float inv_mass_sum = (inv_mass_i + (1.0f / mass[j]));
        float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_rel) / inv_mass_sum;

        //printf("(single) [%u,%u] v_rel=%.2f", i, j, v_rel);
        //printf("(single) [%u,%u] total_mass=%.2f  inv_mass_sum=%.10f", i, j, mass[i] + mass[j], inv_mass_sum);
        //printf("(single) [%u,%u] scalar_impulse_magnitude=%.2f", i, j, scalar_impulse_magnitude);

        if (scalar_impulse_magnitude <= scalar_impulse_magnitude_max) {
            scalar_impulse_magnitude_max = scalar_impulse_magnitude;
            r_norm_max = r_norm;
            j_max = j;
        }
    );
  
    float wg_scalar_impulse_magnitude_max = work_group_reduce_min(scalar_impulse_magnitude_max);
    int candidate = (wg_scalar_impulse_magnitude_max == scalar_impulse_magnitude_max) ? (int)lane : INT_MAX;

    // 1) Find minimum lane index where predicate is true
    int src_lane = work_group_reduce_min(candidate);

    WRITE_LEDGER((i != j_max && lane == src_lane), i, j_max, BOUNCE);
    if (lane == src_lane) {
        COMPLEX final_impulse = r_norm_max * wg_scalar_impulse_magnitude_max * inv_mass_i;
        velocity_intermediate[i] = velocity[i] + final_impulse;
    }
}



__kernel void compute_bouncing_collision_simple(
             const uint    N_bodies_alloc,
             const uint    N_bodies_valid,
    __global const uint*   restrict flags,
    __global const COMPLEX* restrict position,
    __global const COMPLEX* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const COMPLEX* restrict time_until_contact,
    __global const float*  restrict edge_dist,
    __global const float*  restrict v_along_norm,
    __global       COMPLEX* restrict velocity_intermediate
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0) {
        if (lane == 0) {
            velocity_intermediate[i] = velocity[i];
        }
        return;
    }

    float inv_mass_i = 1.0f / mass[i];

    GRID_STRIDE_IJ(

        // We only care about bodies that support bouncing
        if ((flags[j]&BOUNCE_AS_SECONDARY) == 0 || (flags[j]&REMOVED)) continue;
        if (edge_dist[IDX] > EPS_DIST) continue;
        if (v_along_norm[IDX] >= 0) continue;

        float inv_mass_sum = (inv_mass_i + (1.0f / mass[j]));
        float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_along_norm[IDX]) / inv_mass_sum;

        COMPLEX dP = position[j] - position[i];
        COMPLEX r_norm = normalize(dP);
        COMPLEX final_impulse = r_norm * scalar_impulse_magnitude * inv_mass_i;

        velocity_intermediate[i] = velocity[i] + final_impulse;
    );
}