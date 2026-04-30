

__kernel void compute_distance_edge(
               const uint    N,
    __global   const float2* restrict position,
    __global   const float*  restrict radius,
    __global         float*  restrict distance_edge
) {
    uint i = get_group_id(0);
    if (i >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    uint row_start = i * N;
    for (uint j = lane; j < N; j += Lx) {
        if (j == i) continue;
        
        float R = radius[j] + radius[i];
        float dist = distance(position[j], position[i]);
        distance_edge[row_start + j] = dist - R;
    }
}
