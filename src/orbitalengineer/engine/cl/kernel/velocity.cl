

__kernel void compute_velocity(
               const uint    N,
               const float   dt,
    __global   const float2* restrict acceleration,
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
    
    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        if (j == i) continue;
        dV_accum += distance_edge[row_start + j] > 0 ? acceleration[row_start + j] : 0.0f;
    }
  
    float wg_dVx = work_group_reduce_add(dV_accum.x);
    float wg_dVy = work_group_reduce_add(dV_accum.y);
    
    if (lane == 0) { 
        velocity[i] += (float2)(wg_dVx, wg_dVy);
    }
}
