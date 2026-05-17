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
    __global       float*  restrict toi,
    __global       float*  restrict node_dt
) {
    GRID_STRIDE_INIT();

    float min_impact = INFINITY;
    GRID_STRIDE_IJ(
        float R = radius[j] + radius[i];
        cfloat dV = velocity[j] - velocity[i];
        cfloat dP = position[j] - position[i];

        float toi_ij = compute_time_of_impact(dV, dP, R);
        toi[IDX] = toi_ij;
        min_impact = fmin(min_impact, toi_ij);
    );

    float wg_min_impact = work_group_reduce_min(min_impact);
    if (lane == 0)
        node_dt[i] = wg_min_impact;
}


__kernel void assign_interaction_groups(
               const uint    N,
               const float   dt,
    __global   const float*  restrict mass,
    __global   const float*  restrict toi,
    __global         uint*   restrict interaction_group
) {
    GRID_STRIDE_INIT();
    uint min_index = i;
    GRID_STRIDE_IJ(
        if (mass[j] == 0) continue;
        min_index = (toi[IDX] <= dt) ? min(min_index, j) : min_index;
    );

    uint wg_min_index = work_group_reduce_min(min_index);
    if (lane == 0) {
        interaction_group[i] = wg_min_index;
    }
}

__kernel void reduce_interaction_groups(
               const uint    N,
    __global   const float*  restrict mass,
    __global   const float*  restrict node_dt,
    __global         float*  restrict group_dt,
    __global         uint*   restrict interaction_group,
    __global         uint*   restrict has_updates
) {
    uint i = get_global_id(0);
    if (i >= N || mass[i] == 0) return;

    uint workgroup_id = get_group_id(0);
    uint lane = get_local_id(0);
    if (lane == 0) {
        has_updates[workgroup_id] = false;
    }

    // Walk up however many we can
    // We want the largest group ID for the simultaneous interactions
    uint j = interaction_group[i];
    bool updated = false;
    float min_time = node_dt[i];
    while (j != interaction_group[j] && interaction_group[j] > j) {
        j = interaction_group[j];

        min_time = min(min_time, node_dt[j]);
        updated = true;
    }
    
    group_dt[i] = min_time;

    interaction_group[i] = j;
    bool wg_has_updated = work_group_any(updated);
    
    if (lane == 0) {
        has_updates[workgroup_id] = wg_has_updated;
    }
}

__kernel void collect_group_members(
             const uint  N,
    __global const uint* restrict interaction_group,
    __global        int* restrict group_members
) {
    uint i = get_global_id(0);
    if (i >= N) return;
 
    uint group_idx = i;
    uint  row_start = i * N;
    for (uint j = 0; j < N; j += 1) {
        uint const IDX = row_start + j; 
        group_members[group_idx * N] = (interaction_group[j] == i) ? j : -1;
    };
}
