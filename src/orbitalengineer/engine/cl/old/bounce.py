import pyopencl as cl

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
#include "complex.clh"
#include "pair.clh"
#include "force.clh"
#include "scalar_impulse_magnitude.clh"

#ifndef G
#define G 1.0f
#endif

__kernel void bounce_pairwise(
               const float   dt,
    __constant       Pair*   pair,
    __global   const cfloat* position,
    __global   const cfloat* velocity,
    __global   const float*  mass,
    __global   const float*  radius,
    __global         float*  toi,
    __global         cfloat* pair_vel_ij,
    __global         cfloat* pair_vel_ji
) {
    uint g0 = get_global_id(0);
    uint i = pair[g0].i;
    uint j = pair[g0].j;
        
    if (toi[g0] == 0) {
        float inv_mass_i = 1.0 / mass[i];
        float inv_mass_j = 1.0 / mass[j];
        
        cfloat dP = position[j] - position[i];
        cfloat r_norm = normalize(dP);
        float impulse = scalar_impulse_magnitude(i, j, r_norm, velocity, mass);
        pair_vel_ij[g0] +=  r_norm * (impulse * inv_mass_i);
        pair_vel_ji[g0] += -r_norm * (impulse * inv_mass_j);
    }
}
"""

class BouncePairwise(KernelBuilder):
    kernel_name = "bounce_pairwise"
    kernel_src = kernel_src

    def __call__(
        self,
        num_pairs: int, 
        dt_step: float,
        pairs: cl.Buffer,
        pos: cl.Buffer,
        vel: cl.Buffer,
        mass: cl.Buffer,
        radius: cl.Buffer,
        toi: cl.Buffer,
        pair_vel_ij: cl.Buffer,
        pair_vel_ji: cl.Buffer,
        
        metric_alias:str|None=None
    ):
        return self.kernel(
            self.queue,
            (num_pairs,),
            None,
            
            # Args
            dt_step, pairs,
            pos, vel, mass, radius,
            toi, pair_vel_ij, pair_vel_ji,
            
            metric_alias=metric_alias
        )
