#include "kernel/complex.clh"
#include "kernel/stride.clh"

inline float compute_time_of_impact(const cfloat dV, const cfloat dP, const float R) {
    // Coefficients
    float a = dot(dV, dV);
    float b = 2.0f * dot(dP, dV);
    float c = dot(dP, dP) - R * R;

    // Already inside or touching the interaction radius
    if (c <= 0.0f) return 0.0f;

    // No relative motion -> never enters in the future
    const float eps = 1e-12f;
    if (a <= eps) return INFINITY;

    // Discriminant
    float D = b * b - 4.0f * a * c;

    if (a <= eps || D < 0.0f || !isfinite(D))
        return INFINITY;

    float sqrtD = sqrt(D);
    float inv2a = 0.5f / a;
    float t1 = (-b - sqrtD) * inv2a;
    float t2 = (-b + sqrtD) * inv2a;

    return (t1 >= 0.0f) ? t1 : ((t1 >= 0.0f) ? t2 : INFINITY);
}

__kernel void compute_interaction_time(
             const uint    N,
    __global const cfloat* restrict position,
    __global const cfloat* restrict velocity,
    __global const float*  restrict radius,
    __global       float*  restrict impact_times,
    __global       float*  restrict min_impact_time
) {
    GRID_STRIDE_INIT();

    float min_impact = INFINITY;
    GRID_STRIDE_IJ(
        float R = radius[j] + radius[i];
        cfloat dV = velocity[j] - velocity[i];
        cfloat dP = position[j] - position[i];

        float toi_ij = compute_time_of_impact(dV, dP, R);
        impact_times[IDX] = toi_ij;
        min_impact = fmin(min_impact, toi_ij);
    );

    float wg_min_impact = work_group_reduce_min(min_impact);
    if (lane == 0)
        min_impact_time[i] = wg_min_impact;
}


__kernel void assign_interaction_groups(
               const uint    N,
               const float   dt,
    __global   const float*  restrict mass,
    __global   const float*  restrict impact_times,
    __global         uint*   restrict interaction_group
) {
    GRID_STRIDE_INIT();

    uint min_index = i;
    GRID_STRIDE_IJ(
        if (mass[j] == 0) continue;
        if (impact_times[IDX] <= dt)
            min_index = min(min_index, j);
    );

    uint wg_min_index = work_group_reduce_min(min_index);
    if (lane == 0) {
        interaction_group[i] = wg_min_index;
    }
}

__kernel void reduce_interaction_groups(
               const uint    N,
    __global   const float*  restrict mass,
    __global         uint*   restrict interaction_group,
    __global         uint*   restrict has_updates
) {
    uint i = get_global_id(0);

    // Get own interaction_group ID. If it's itself, then we're done!
    if (i >= N || mass[i] == 0 || i == interaction_group[i]) return;

    uint workgroup_id = get_group_id(0);
    uint lane = get_local_id(0);
    if (lane == 0) {
        has_updates[workgroup_id] = false;
    }

    // Walk up however many we can
    // We want the largest group ID for the simultaneous interactions
    uint j = interaction_group[i];
    bool updated = false;
    while (j != interaction_group[j] && interaction_group[j] > j) {
        j = interaction_group[j];
        updated = true;
    }
    
    interaction_group[i] = j;
    bool wg_has_updated = work_group_any(updated);
    if (lane == 0) {
        has_updates[workgroup_id] = wg_has_updated;
    }
}


/*
      IDX | 00  01  02  03 | 04  05  06  07 | 08  09  10  11 | 12  13  14  15 | 16  17  18  19 |
  MIN TOI | 01  01  03  11 | 13  22  19  44 | 54  31  67  21 | 42  05  31  12 | 22  41  12  10 |
    GROUP | 00  01  04  04 | 04  05  06  07 | 08  09  10  00 | 00  06  14  15 | 01  17  19  19 |

GROUPS: 00, 01, 04, 05, 06, 07, 08, 09, 10, 14, 15, 17, 19

GROUP MEMBERS:

      IDX       |  00  01  02  03  04  05
  --------------+---------------------------
  Group 00 (00) |  00  11  12
  Group 01 (01) |  01  16
  Group 02 (04) |  02  03  04
  Group 03 (05) |  05
  Group 04 (06) |  06  13
  Group 05 (07) |  07
  Group 06 (08) |  08
  Group 07 (09) |  09
  Group 08 (10) |  10
  Group 09 (14) |  14
  Group 10 (15) |  15
  Group 11 (17) |  17
  Group 12 (19) |  18  19



 MIN GTOI | 01  01  13  13


*/
__kernel void compute_interaction_group_min_time(
               const uint    N,
    __global   const float*  restrict mass,
    __global   const uint*   restrict interaction_group,
    __global   const float*  restrict min_node_impact_time,
    __global         float*  restrict min_group_impact_time
) {
    GRID_STRIDE_INIT();
    GRID_STRIDE_IJ(
        if (mass[j] == 0) continue;
        if (impact_times[IDX] <= dt)
            min_index = min(min_index, j);
    );

    uint wg_min_index = work_group_reduce_min(min_index);
    if (lane == 0) {
        interaction_group[i] = wg_min_index;
    }
}