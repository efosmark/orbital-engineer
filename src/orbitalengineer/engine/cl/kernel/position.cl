

__kernel void compute_position(
               const float   dt,
    __global   const float2* restrict velocity,
    __global         float2* restrict position
) {
    uint i = get_global_id(0);
    position[i] += velocity[i] * dt;
}