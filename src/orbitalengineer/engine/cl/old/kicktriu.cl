#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#include "complex.clh"
#include "toi.clh"
#include "force.clh"
#include "pair.clh"
#include "status.clh"

__kernel void compute_impact_times(
    __constant Pair*   pair,
    __global   float*  mass,
    __global   cfloat* position,
    __global   cfloat* velocity,
    __global   float*  radius,
    __global   cfloat* p_part,
    __global   cfloat* v_part,
    __global   uint*   status,
    __global   float*  toi,
    __global   float*  overlap
    //__global   int*    group_part
) {
    uint ij = get_global_id(0);
    uint i = pair[ij].i;
    uint j = pair[ij].j;

    cfloat dV = velocity[j] - velocity[i]; // Relative velocity
    cfloat dP = position[j] - position[i]; // Relative position
    float R = radius[j] + radius[i];       // Sum of radii
    float dist = fast_distance(position[j], position[i]);

    // Coefficients
    float a = dot(dV, dV);
    float b = 2.0f * dot(dP, dV);
    float c = dot(dP, dP) - R*R;

    // Discriminant
    // * D < 0 → no real solution → no collision ever
    // * D = 0 → spheres graze exactly once → tangent collision
    // * D > 0 → two roots → approach then separate
    float D = discriminant(a, b, c);

    // Compute the roots (t1, t2)
    float2 q = solve_quadratic(a, b, D);

    // Distance from edge-to-edge
    float edge_dist = dist - R;
    uint is_approaching = D > 0 && q.s0 >= 0;

    toi[ij] = INFINITY;
    status[ij] = c <= 0 ? STATUS_OVERLAP : STATUS_NOMINAL;
 

    uint idx_i = pair[ij].idx_i;
    uint idx_j = pair[ij].idx_j;

    v_part[idx_i] = 0;
    v_part[idx_j] = 0;

    p_part[idx_i] = 0;
    p_part[idx_j] = 0;
    
    cfloat r_norm = normalize(dP);
    float inv_mass_i = 1.0 / mass[i];
    float inv_mass_j = 1.0 / mass[j];
    float inv_mass_sum = (inv_mass_i + inv_mass_j);
    
    if (status[ij] == STATUS_OVERLAP) {
        float k = edge_dist / inv_mass_sum;
        p_part[idx_i] += r_norm * (k * inv_mass_i);
        p_part[idx_j] -= r_norm * (k * inv_mass_j);

        status[ij] = STATUS_TOUCH;
        toi[ij] = 0;
    }

    // else if (status[ij] == STATUS_TOUCH) {
    //     float impulse = scalar_impulse_magnitude(i, j, r_norm, velocity, mass);
    //     v_part[idx_i] =  r_norm * (impulse * inv_mass_i);
    //     v_part[idx_j] = -r_norm * (impulse * inv_mass_j);
    // }
}


// __kernel void resolve_overlap_position(
//     __constant Pair*   pair,
//     __global   float*  mass,
//     __global   float*  radius,
//     __global   cfloat* position,
//     __global   uint*   status,
//     __global   float*  toi,
//     __global   cfloat* p_part
// ) {
//     uint ij = get_global_id(0);

//     uint i = pair[ij].i;
//     uint j = pair[ij].j;

//     uint idx_i = pair[ij].idx_i;
//     uint idx_j = pair[ij].idx_j;    

//     p_part[idx_i] = 0;
//     p_part[idx_j] = 0;
    
//     if (status[ij] == STATUS_OVERLAP) {
        
//         cfloat dP = position[j] - position[i];                 // Relative position
//         cfloat r_norm = normalize(dP);
        
//         float inv_mass_i = 1.0 / mass[i];
//         float inv_mass_j = 1.0 / mass[j];
//         float inv_mass_sum = (inv_mass_i + inv_mass_j);
        
//         float R = radius[j] + radius[i];                       // Sum of radii
//         float dist = fast_distance(position[j], position[i]);

//         // Distance from edge-to-edge
//         float edge_dist = dist - R;

//         float k = edge_dist / inv_mass_sum;
//         p_part[idx_i] += r_norm * (k * inv_mass_i);
//         p_part[idx_j] -= r_norm * (k * inv_mass_j);

//         status[ij] = STATUS_TOUCH;
//         toi[ij] = -INFINITY;
//     }
// }



/*

TIME OF INTERACTION
-------------------

table_toi:

[  [ 0.00, 0.35, 0.11, 0.23, 0.56, 0.83 ],
   [ 0.35, 0.00, 0.16, 0.23, 0.72, 0.29 ],
   [ 0.11, 0.35, 0.00, 0.50, 0.61, 0.31 ],
   [ 0.23, 0.16, 0.50, 0.00, 0.87, 0.99 ],
   [ 0.56, 0.72, 0.61, 0.87, 0.00, 0.15 ],
   [ 0.83, 0.29, 0.31, 0.99, 0.15, 0.00 ]  ]

toi_min:
[ 0.11, 0.16, 0.11, 0.16, 0.15, 0.15 ]
   

GROUPING (dt=0.25)
------------------

table_grouping
[  [ --,  0,  0,  0,  0,  0 ],
   [  1, --,  1,  1,  1,  1 ],
   [  0,  2, --,  2,  2,  2 ],
   [  0,  1,  3, --,  3,  3 ],
   [  4,  4,  4,  4, --,  4 ],
   [  5,  5,  5,  5,  4, -- ]  ]

grouping_min
[ 0, 1, 0, 0, 4, 4  ]

*/