#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/complex.clh" // needed for creal, cmul, cconj

inline float relative_speed_along_normal(
    const uint i,
    const uint j,
    const float2 r_norm,
    const float2* velocity
) {
    float2 vr = velocity[i] - velocity[j]; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}


__kernel void compute_center_of_mass(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const uint*   restrict ids,
    __global const uint*   restrict num_contacts,
    __global const uint*   restrict direct_contacts,
    __global       float2* restrict mv_com,
    __global       float2* restrict v_cm,
    __global       float*  restrict mass_cm
) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);

    uint i = ids[gid];
    //bool i_enabled = true; //(flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0;
    //if (!i_enabled) return;

    if (lid >= num_contacts[i]) return;

    // Get the individual collision
    uint j = direct_contacts[(N * i) + lid];
    if (j == i) return;
    //bool j_enabled = true; //(flags[j]&REMOVED) || (flags[j]&BOUNCE_AS_SECONDARY) == 0;
    //if (!j_enabled) return;

    float2 mr_j = position[j] * mass[j];
    float2 mr_contact = FLOAT2_WG_REDUCE_ADD(mr_j);
    float2 mr_self = position[i] * mass[i];
    float2 mr_total = mr_contact + mr_self;
    
    float2 mv_j = velocity[j] * mass[j];
    float2 mv_contact = FLOAT2_WG_REDUCE_ADD(mv_j);
    float2 mv_self = velocity[i] * mass[i];
    float2 mv_total = mv_contact + mv_self;
    
    float m_contact = work_group_reduce_add(mass[j]);
    float m_self = mass[i];
    float m_total = m_contact + m_self;
    
    if (lid == 0) {
        
        mv_com[i] = mv_total;
        //p_total[i] = mv_total;
        v_cm[i] = mv_total / m_total;
        mass_cm[i] = m_total;

        //printf("(com) [%u] velocity = %.2f %.2f", i, v_cm[i].x, v_cm[i].y);
        //printf("(com) [%u] mass = %.2f", i, mass_cm[i]);
    }
}


__kernel void compute_impulse(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const uint*   restrict ids,
    __global const uint*   restrict num_contacts,
    __global const uint*   restrict direct_contacts,
    __global const float2* restrict mv_com,
    __global const float2* restrict v_cm,
    __global const float*  restrict mass_cm,
    __global       float2* restrict impulse_out
) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);

    uint i = ids[gid];
    //bool i_enabled = true; //(flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0;
    //if (!i_enabled) return;

    if (lid >= num_contacts[i]) return;

    // Get the individual collision
    uint j = direct_contacts[(N * i) + lid];
    if (j == i) return;

    //bool j_enabled = true; //(flags[j]&REMOVED) || (flags[j]&BOUNCE_AS_SECONDARY) == 0;
    //if (!j_enabled) return;

    //float2 vj = velocity[j];
    //float2 total_vj = FLOAT2_WG_REDUCE_ADD(vj);
    //float2 total_v = velocity[i] + total_vj;

    //float2 vr = velocity[i] - velocity[j]; // relative speed
    //float2 total_vr = FLOAT2_WG_REDUCE_ADD(vr);


    //float2 ri_cm = position[i] - r_cm[i];
    //float2 rj_cm = position[j] - r_cm[i];
    float2 r_norm = normalize(position[j] - position[i]);


    float2 mv_total = mv_com[i] + mv_com[j];

    // geometric mean between the two CoM frames
    float m_total = (mass_cm[i] + mass_cm[j]);


    float2 velocity_com = mv_total / m_total;

    // our speed in the com frame
    float2 vi_cm = velocity[i] ;//- velocity_com;
    float2 vj_cm = velocity[j] ;//- velocity_com;

    float2 vdiff = vj_cm - vi_cm;
    //printf("(compute) [%u,%u] vdiff = %.2f %.2f", i, j, vdiff.x, vdiff.y);


    //float2 total_r_norm = normalize(FLOAT2_WG_REDUCE_ADD(r_norm));

    //float inv_mass_i = 1.0f / mass[i];
    //float inv_mass_j = 1.0f / mass[j];
    //float total_contact_inv_mass = work_group_reduce_add(1.0f / mass_j);
    //float total_inv_mass = inv_mass_i + total_contact_inv_mass;
    //float total_v_rel = creal(cmul(total_vr, cconj(total_r_norm))); // along normal (scalar)
    
    //float total_contact_mass = work_group_reduce_add(mass_j);
    //float total_mass = mass[i] + total_contact_mass;

    //float2 total_contact_momentum = FLOAT2_WG_REDUCE_ADD(mass[j] * velocity[j]);
    //float2 total_momentum = (mass[i] * velocity[i]) + total_momentum;
    //float2 com_mv_j = (velocity[j] * mass[j]) / mass_cm[j];
    //float2 com_mv_i = (velocity[i] * mass[i]) / mass_cm[i];

    //float2 combined_com = v_cm[i] + v_cm[j]

    //float2 vdiff = ((v_cm[j] - com_mv_i)/ mass_cm[i] - (v_cm[i] - com_mv_j)/ mass_cm[i]) ;
    //float2 vdiff = velocity[j] - velocity[i];
    //float2 vdiff = v_cm[j]/ - v_cm[i];



    float v_rel_along_norm = creal(cmul(vdiff, cconj(r_norm)));
    if (v_rel_along_norm >= 0) return;

    //printf("(compute) [%u,%u] v_rel_along_norm = %.2f", i, j, v_rel_along_norm);
    //printf("(compute) [%u,%u] total_inv_mass   = %.10f", i, j, total_inv_mass);


    //float2 v = (total_momentum + (mass[i] * (1.0f + COEF_OF_RESTITUTION) * vdiff)) / total_mass;

    float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_rel_along_norm) * (mass[j]);

    float2 v = r_norm * scalar_impulse_magnitude / (mass[j] + mass[i]);

    //v_rel = relative_speed_along_normal(i, j, r_norm, velocity);
    //if (v_rel <= 0) return;

    //float mv_rel = v_rel * mass[i];

    //printf("(compute) [%u,%u] total_v_rel=%.2f", i, j, total_v_rel);
    //printf("(compute) [%u,%u] mv_rel=%.2f", i, j, mv_rel);

    //float total_contact_mass = work_group_reduce_add(mass_j);
    //float total_momentum = work_group_reduce_add(mv_rel);

    //printf("(compute) [%u,%u] wg_inv_mass_sum=%.10f", i, j, wg_inv_mass_sum);
    
    //float scalar_impulse_magnitude = (((2.0f + COEF_OF_RESTITUTION) * total_v_rel) / (inv_mass_i + total_inv_mass));
    
    //printf("(compute) [%u,%u] scalar_impulse_magnitude=%.10f", i, j, scalar_impulse_magnitude);
    //printf("(compute) [%u,%u] r_norm=%.2f,%.2f", i, j, r_norm.x, r_norm.y);
    

    //float total_scalar_impulse_magnitude = work_group_reduce_add(scalar_impulse_magnitude);

    //scalar_impulse_magnitude *= (inv_mass_j)/total_inv_mass;

    //float2 final_impulse = (r_norm * scalar_impulse_magnitude) / (mass_i+mass_j) ;//  *   (mass_j/(total_mass+mass_i));
    float2 final_impulse = v;
    //float2 final_impulse = (float2)(-v.x, -v.y) * r_norm;

    // Set the impulse_out in THEIR row
    uint idx = (N * i) + j;
    impulse_out[idx] = final_impulse;    
    //if (impulse_out[idx].x != 0 || impulse_out[idx].y != 0)
    //    printf("(compute) impulse[row:%u,col:%u] = %.2f %.2f", j, i, impulse_out[idx].x, impulse_out[idx].y);
}


__kernel void assign_impulse(
             const uint    N,
    __global const uint*   restrict ids,
    __global const uint*   restrict num_contacts,
    __global const uint*   restrict direct_contacts,
    __global const float2* restrict impulse,
    __global       float2* restrict velocity_intermediate
) {
    uint g0 = get_group_id(0);
    uint lid = get_local_id(0);
    uint Lx = get_local_size(0);

    uint i = ids[g0];
    if (lid >= num_contacts[i]) return;

    uint j = direct_contacts[(N * i) + lid];

    uint idx = (N * i) + j;
    float2 impulse_from_j = impulse[idx];
    
    //if (impulse_from_j.x != 0 || impulse_from_j.y != 0)
    //    printf("(assign ) impulse[row:%u,col:%u] = %.2f %.2f", i, j, impulse_from_j.x, impulse_from_j.y);

    float2 wg_total_impulse = FLOAT2_WG_REDUCE_ADD(impulse_from_j);
    if (lid == 0) {

        //if (wg_total_impulse.x != 0 || wg_total_impulse.y != 0)
        //    printf("(assign) [%u] wg_total_impulse = %.2f %.2f", i, wg_total_impulse.x, wg_total_impulse.y);
        velocity_intermediate[i] += wg_total_impulse;
    }
}


__kernel void compute_bouncing_collision_single(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const float2* restrict time_until_contact,
    __global       float2* restrict velocity_intermediate
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0) {
        if (lane == 0) {
            velocity_intermediate[i] = velocity[i];
        }
        return;
    }

    float scalar_impulse_magnitude_max = 0.0f;
    float2 r_norm_max = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];

    GRID_STRIDE_IJ(

        // We only care about bodies that support bouncing
        if ((flags[j]&BOUNCE_AS_SECONDARY) == 0 || (flags[j]&REMOVED)) continue;

        // Limit to only contacting bodies
        float2 toi = time_until_contact[IDX];
        float t1 = toi.x;
        float t2 = toi.y;
        float curr_min = (t1 >= 0.0f) ? t1 : ((t2 >= 0.0f) ? t2 : EPS_TIME);
        if (curr_min > EPS_TIME) continue;

        float2 dP = position[j] - position[i];
        float edge_dist = fast_length(dP) - radius[i] - radius[j];
        if (edge_dist > EPS_DIST) continue;

        float2 r_norm = normalize(dP);

        float v_rel = relative_speed_along_normal(i, j, -r_norm, velocity);
        if (v_rel >= 0) continue;

        // float edge_dist = fast_length(position[j] - position[i]) - radius[i] - radius[j];
        // bool is_touching = edge_dist <= EPS_DIST;
        // if (!is_touching) continue;

        ///printf("(%u, %u)", i, j);

        float inv_mass_sum = (inv_mass_i + (1.0f / mass[j]));
        float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_rel) / inv_mass_sum;


        //printf("(single) [%u,%u] v_rel=%.2f", i, j, v_rel);
        //printf("(single) [%u,%u] total_mass=%.2f  inv_mass_sum=%.10f", i, j, mass[i] + mass[j], inv_mass_sum);
        //printf("(single) [%u,%u] scalar_impulse_magnitude=%.2f", i, j, scalar_impulse_magnitude);

        //if (i == 2) printf("(%u, %u)  |si|=%.2f", i, j, scalar_impulse_magnitude);
        if (scalar_impulse_magnitude <= scalar_impulse_magnitude_max) {
            scalar_impulse_magnitude_max = scalar_impulse_magnitude;
            r_norm_max = r_norm;
        }
    );
  
    float wg_scalar_impulse_magnitude_max = work_group_reduce_min(scalar_impulse_magnitude_max);
    //if (wg_scalar_impulse_magnitude_max > 0) printf("wg_scalar_impulse_magnitude_max=%u", wg_scalar_impulse_magnitude_max);

    int candidate = (wg_scalar_impulse_magnitude_max == scalar_impulse_magnitude_max) ? (int)lane : INT_MAX;

    // 1) Find minimum lane index where predicate is true
    int src_lane = work_group_reduce_min(candidate);

    // If src_lane == INT_MAX, no lane satisfied the predicate
    //bool any = (src_lane != INT_MAX);

    // // 2) Let only that lane supply a meaningful value
    // float my_value = ...;
    // float value_to_broadcast = (lane == (uint)src_lid) ? my_value : 0.0f;

    // // Optional: you may want to guard if !any (no candidates)
    // float b = any
    //     ? work_group_broadcast(value_to_broadcast, (uint)src_lid)
    //     : 0.0f;  // or some neutral element

    //float2 wg_max_impulse = FLOAT2_WG_REDUCE_MAX(scalar_impulse_magnitude_max);
    //if (i > 0 && src_lane != 0) {
    //    printf("i=%u  src_lane=%u", i, src_lane);
    //}
    
    if (lane == src_lane) {
        //if (i == 2) printf("(%u) |si|_wg=%.2f  lane=%u  src_lane=%u", i, wg_scalar_impulse_magnitude_max, lane, src_lane);
        //if (wg_scalar_impulse_magnitude_max > 0) printf("%u %u", i, scalar_impulse_magnitude_max);
        float2 final_impulse = r_norm_max * wg_scalar_impulse_magnitude_max * inv_mass_i;
        
        velocity_intermediate[i] = velocity[i] + final_impulse;
        //if (i == 0) {
            //printf("(%u) velocity         = (%.1f,%.1f)", i, velocity[i].x, velocity[i].y);
            //printf("(%u) final_impulse    = (%.1f,%.1f)", i, final_impulse.x, final_impulse.y);
            //printf("(%u) vel_intermediate = (%.1f,%.1f)", i, velocity_intermediate[i].x, velocity_intermediate[i].y);
        //}
    }
}



__kernel void compute_bouncing_collision_simple(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global const float2* restrict time_until_contact,
    __global const float*  restrict edge_dist,
    __global const float*  restrict v_along_norm,
    __global       float2* restrict velocity_intermediate
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

        // Limit to only contacting bodies
        //float2 toi = time_until_contact[IDX];
        //float t1 = toi.x;
        //float t2 = toi.y;
        //float curr_min = (t1 > 0.0f) ? t1 : ((t2 > 0.0f) ? t2 : INFINITY);
        //if (curr_min > EPS_TIME) continue;

        if (edge_dist[IDX] > EPS_DIST) continue;
        if (v_along_norm[IDX] >= 0) continue;


        float inv_mass_sum = (inv_mass_i + (1.0f / mass[j]));
        float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_along_norm[IDX]) / inv_mass_sum;


        float2 dP = position[j] - position[i];
        float2 r_norm = normalize(dP);
        float2 final_impulse = r_norm * scalar_impulse_magnitude * inv_mass_i;

        velocity_intermediate[i] = velocity[i] + final_impulse;
    );
}