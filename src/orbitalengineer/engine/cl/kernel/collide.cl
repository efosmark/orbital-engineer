

__kernel void compute_collision(
               const uint    N,
    __global   const float*  restrict velocity_relative,
    __global   const float2* restrict position,
    __global         float2* restrict velocity,
    __global   const float*  restrict mass,
    __global   const float*  restrict distance_edge
) {
    uint i = get_group_id(0);
    if (i >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    float2 dV_accum = (float2)(0.0f, 0.0f);
    float inv_mass_i = 1.0f / mass[i];
    float2 pos_i = position[i];

    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        
        float v_rel = velocity_relative[row_start + j];
        bool is_touching = distance_edge[row_start + j] <= 0.0f;
        if (j == i || v_rel >= 0 || !is_touching) continue;

        float inv_mass_j = 1.0f / mass[j];
        float inv_mass_sum = (inv_mass_i + inv_mass_j);
        
        // scalar impulse magnitude
        float2 dP = position[j] - pos_i;
        float2 r_norm = normalize(dP);
        float impulse = ((1.0f + COEF_OF_RESTITUTION) * v_rel) / inv_mass_sum;
        dV_accum += (r_norm * (impulse * inv_mass_i));
    }
  
    float wg_dVx = work_group_reduce_add(dV_accum.x);
    float wg_dVy = work_group_reduce_add(dV_accum.y);

    if (lane == 0) { 
        velocity[i] += (float2)(wg_dVx, wg_dVy);
    }
}
