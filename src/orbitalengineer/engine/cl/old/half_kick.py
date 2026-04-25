import numpy as np
import pyopencl as cl

from orbitalengineer.engine.cl.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#include "complex.clh"
#include "pair.clh"
#include "status.clh"
#include "debug.clh"

#ifndef G
#define G 1.0f
#endif

inline float relative_speed_along_normal(uint i, uint j, cfloat r_norm, cfloat* velocity) {
    cfloat vr = velocity[i] - velocity[j]; // relative speed
    return creal(cmul(vr, cconj(r_norm))); // along normal (scalar)
}

inline float scalar_impulse_magnitude(uint i, uint j, cfloat r_norm, cfloat* velocity, float* mass) {
    float vn = relative_speed_along_normal(i, j, -r_norm, velocity);

    // if already separating, no response
    vn = select(vn, (float)(0), vn > 0.0);

    float inv_mass_i = 1.0 / mass[i];
    float inv_mass_j = 1.0 / mass[j];
    float inv_mass_sum = (inv_mass_i + inv_mass_j);

    // scalar impulse magnitude
    return ((1.0 + COEF_OF_RESTITUTION) * vn) / inv_mass_sum;
}

inline cfloat force(
    const int i,
    const int j,
    const cfloat dr,
    const float dist,
    const float *mass
) {
    return ((G * mass[i] * mass[j]) / (dist*dist*dist)) * dr;
}

__kernel void half_kick_pairwise(
//             float   dt,
//             uint    N,
    __constant Pair*   pair,
    __global   uint*   status,
    __global   cfloat* position,
    __global   cfloat* velocity,
    __global   float*  mass,
    __global   float*  radius,
    __global   float*  toi,
    __global   float*  group_part,
    __global   cfloat* v_part
) {
    
    uint pairwise_idx = get_global_id(0);
    uint i = pair[pairwise_idx].i;
    uint j = pair[pairwise_idx].j;
    
    uint idx_i = pair[pairwise_idx].idx_i;
    uint idx_j = pair[pairwise_idx].idx_j;

    v_part[idx_i] = 0;
    v_part[idx_j] = 0;

    
    cfloat dr = position[j] - position[i];
    
    if (status[pairwise_idx] == STATUS_TOUCH) {
        cfloat r_norm = normalize(dr);
        float inv_mass_i = 1.0 / mass[i];
        float inv_mass_j = 1.0 / mass[j];
        
        float impulse = scalar_impulse_magnitude(i, j, r_norm, velocity, mass);
        v_part[idx_i] +=  r_norm * (impulse * inv_mass_i);
        v_part[idx_j] += -r_norm * (impulse * inv_mass_j);
    }
    
    float dist = fast_length(dr);
    
    // Apply Netwon's theory of universal gravitation
    cfloat f_ij = force(i, j, dr, dist, mass) * dt;
    v_part[idx_i] +=  f_ij / mass[i];
    v_part[idx_j] += -f_ij / mass[j];
}
"""


class HalfKickPairwise(KernelBuilder):
    kernel_name = "half_kick_pairwise"
    kernel_src = kernel_src

    def __call__(
        self,
        N: int,
        num_pairs: int, 
        dt_step: float,
        pairs: cl.Buffer,
        status: cl.Buffer,
        pos: cl.Buffer,
        vel: cl.Buffer,
        mass: cl.Buffer,
        radius: cl.Buffer,
        toi: cl.Buffer,
        group_part: cl.Buffer,
        vel_pairwise: cl.Buffer
    ):
        return self.kernel(
            self.queue,
            (num_pairs,),
            None,
            
            # Args
            dt_step,
            np.uint32(N),
            pairs,
            status,
            pos,
            vel,     
            mass,
            radius,
            toi, 
            group_part,
            vel_pairwise,
        )