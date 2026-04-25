#pragma OPENCL EXTENSION cl_khr_subgroups : enable

inline float2 bounce(
    const int i,
    const int j,
    const float e,
    const float2 dr,
    const float dist,
    __global const float *mass,
    __global const float2 *position,
    __global const float2 *velocity
) {
    // unit normal from j -> i
    float2 n = dr / dist;

    float2 rv = velocity[j] - velocity[i];

    // relative speed along normal (scalar)
    float2 n_conj = (float2)(n.x - n.y);
    float vn = (rv * n_conj).x;

    // if already separating, no response
    if (vn >= 0.0) n = 0;

    float inv_mass_sum = (1.0 / mass[i]) + (1.0 / mass[j]);

    // scalar impulse magnitude
    float impulse = -(1.0 + e) * vn / inv_mass_sum;

    // apply along normal (complex direction n)
    return (impulse / mass[i]) * n;
    //velocity[j] -= (impulse / mass[j]) * n
}

inline float2 accel(
    const int row,
    const int col,
    const float2 dr,
    const float dist,
    __global const float2 *pos,
    __global const float *mass
) {

    // Newton's Law of Universal Gravitation
    float mu = -1.0 * mass[row] * mass[col];
    float2 f = select(0.0f, mu / (dist*dist*dist), dist > 1) * dr;
    
    // Acceleration
    return f / mass[row];
}


__kernel void compute_acceleration(
    const uint            N,
    const float           dt,
    __global const float2 *position,
    __global const float2 *velocity,
    __global const float  *mass,
    __global const float  *radius,
    __global float2       *restrict partials
) {
    uint tile_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint Lx  = get_local_size(0);
    
    uint row = get_global_id(1);
    uint col = (tile_id * Lx) + local_id;
    
    float2 A;
    
    float2 dr = position[row] - position[col];
    float dist = fast_length(dr) + FLT_EPSILON;
    if (dist <= radius[row] + radius[col]) {
        A = bounce(row, col, 0.99, dr, dist, mass, position, velocity);
        //A -= bounce(col, row, 0.95, dr, dist, mass, position, velocity);
        A *= dt;
    }
    else
        A = accel(row, col, dr, dist, position, mass) * dt;
    
    float accel_x = work_group_reduce_add(A.x);
    float accel_y = work_group_reduce_add(A.y);
    
    if (local_id == 0) {
        uint row_tile_id = (row * Lx) + tile_id;
        partials[row_tile_id].x = accel_x;
        partials[row_tile_id].y = accel_y;
    }
 }


__kernel void apply_acceleration(
    const uint                N,
    const uint                Lx,
    __global const float2     *partials,
    __global float2  *restrict vel
) {
    uint l0 = get_local_id(0);
    
    // Since we are summing for arbitrarily wide rows, each row can be
    // identified by its tile ID (i.e. one tile per row).
    uint row = get_group_id(0);

    // Alternatively, number of WGs along X
    //uint ntiles_per_row = get_num_groups(0);
    //uint row = g0 / ntiles_per_row;
    
    float2 accum = (float2) (0, 0);
    for (uint t = l0; t < Lx; t += 1) {
        uint idx = (row * Lx) + t;
        accum += partials[idx];
    }

    float sum_x = work_group_reduce_add(accum.x);
    float sum_y = work_group_reduce_add(accum.y);

    if (l0 == 0) {
        vel[row] += (float2)(sum_x, sum_y);
    }
}


__kernel void update_position(
             const float       dt,
    __global const float2      *vel,
    __global float2  *restrict pos
) {
    uint i = get_global_id(0);
    pos[i] += vel[i] * dt;
}
