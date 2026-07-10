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


__kernel void compute_bouncing_collision(
             const uint    N,
    __global const uint*   restrict flags,
    __global const float2* restrict position,
    __global const float2* restrict velocity,
    __global const float*  restrict mass,
    __global const float*  restrict radius,
    __global       float2* restrict velocity_intermediate,
    __global       float2* restrict position_intermediate,
    __global       float2* restrict collision_point
) {
    GRID_STRIDE_INIT();

    if ((flags[i]&REMOVED) || (flags[i]&BOUNCE_AS_PRIMARY) == 0) {
        if (lane == 0) {
            velocity_intermediate[i] = velocity[i];
        }
        return;
    }

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];
    float2 pos_i = position[i];
    float2 vel_i = velocity[i];
    float radius_i = radius[i];

    GRID_STRIDE_IJ(
        if ((flags[j]&BOUNCE_AS_SECONDARY) == 0) continue;

        float2 dP = position[j] - pos_i;
        float2 r_norm = normalize(dP);

        float v_rel = relative_speed_along_normal(i, j, -r_norm, velocity);
        float edge_dist = fast_length(position[j] - position[i]) - radius[i] - radius[j];
        bool is_touching = edge_dist <= EPS_DIST;
        if (!is_touching || v_rel >= 0 || (flags[j]&REMOVED) ) continue;
        
        float2 dV = vel_i - velocity[j];

        float inv_mass_sum = (inv_mass_i + (1.0f / mass[j]));
        float scalar_impulse_magnitude = ((1.0f + COEF_OF_RESTITUTION) * v_rel) / inv_mass_sum;
        float2 impulse = (r_norm * (scalar_impulse_magnitude * inv_mass_i));

        dV_accum += impulse;

        // Record the exact location where the two particles are colliding
        //collision_point[IDX] = ((radius_i * position[j]) + (radius[j] * pos_i)) / ((radius[j] + radius_i));

        //#ifdef DEBUG
        //if (i < j) printf("BOUNCE [ %3u, %3u ] @ (%.1f,%.1f)", i, j, collision_point[IDX].x, collision_point[IDX].y);
        //#endif
    );
  
    float2 wg_dV = FLOAT2_WG_REDUCE_ADD(dV_accum);
    if (lane == 0) { 
        velocity_intermediate[i] = velocity[i] + wg_dV;
    }
}
  
