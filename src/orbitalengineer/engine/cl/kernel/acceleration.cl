#ifndef G
#define G 1.0
#endif

inline float2 force(
    const int i,
    const int j,
    const float2 dr,
    const float *mass
) {
    float dist = fast_length(dr);
    float2 mu = ((float)(G) * mass[i] * mass[j]);
    return (mu / (dist*dist*dist)) * dr;
}

__kernel void compute_acceleration(
               const uint    N,
               const float   dt,
    __global   const float2* restrict position,
    __global   const float*  restrict mass,
    __global   const float*  restrict radius,
    __global         float2* restrict acceleration
) {
    uint i = get_group_id(0);
    if (i >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    uint row_start = i * N;
    float inv_mass_i = 1.0f / mass[i];

    for (uint j = lane; j < N; j += Lx) {
        float2 dP = position[j] - position[i];
        acceleration[row_start + j] = force(i, j, dP, mass) * inv_mass_i * dt;
    }
}
