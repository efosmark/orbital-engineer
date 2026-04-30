import pyopencl as cl

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags

"""

    # def kick_nn(self, dt_step):
    #     self.kick_nxn(
    #         self.pairs_host.size,
    #         dt_step,
    #         self.pairs_cl,
    #         self.pos_cl,
    #         self.vel_cl,
    #         self.mass_cl,
    #         self.radius_cl,
    #         self.pair_toi_cl,
    #         self.table_vel_cl,
    #     )
    #     self.cfloat_row_reduce(
    #         self.N,
    #         self.table_vel_cl,
    #         self.vel_cl,
    #         metric_alias="kick_reduce_vel"
    #     )

"""


kernel_src = """
#include "complex.clh"
#include "pair.clh"
#include "force.clh"
#include "scalar_impulse_magnitude.clh"

#ifndef G
#define G 1.0f
#endif

__kernel void kick_nxn(
               const float   dt,
    __constant       Pair*   pair,
    __global   const cfloat* position,
    __global   const cfloat* velocity,
    __global   const float*  mass,
    __global   const float*  radius,
    __global         float*  toi,
    __global         cfloat* v_part
) {
    uint ij = get_global_id(0);
    uint i = pair[ij].i;
    uint j = pair[ij].j;

    cfloat dV = velocity[j] - velocity[i]; // Relative velocity
    cfloat dP = position[j] - position[i]; // Relative position
    float R = radius[j] + radius[i];       // Sum of radii
    float dist = fast_distance(position[j], position[i]);
    
    uint idx_i = pair[ij].idx_i;
    uint idx_j = pair[ij].idx_j;

    float inv_mass_i = 1.0 / mass[i];
    float inv_mass_j = 1.0 / mass[j];
    
    // Apply Netwon's theory of universal gravitation
    cfloat f_ij = force(i, j, dP, dist, mass) * dt;
    v_part[idx_i] =  f_ij * inv_mass_i;
    v_part[idx_j] = -f_ij * inv_mass_j;
        
    if (toi[ij] == 0) {
        cfloat r_norm = normalize(dP);
        float impulse = scalar_impulse_magnitude(i, j, r_norm, velocity, mass);
        v_part[idx_i] +=  r_norm * (impulse * inv_mass_i);
        v_part[idx_j] += -r_norm * (impulse * inv_mass_j);
    }
}
"""

class KickNxN(KernelBuilder):
    kernel_name = "kick_nxn"
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
        vel_table: cl.Buffer
    ):
        return self.kernel(
            self.queue,
            (num_pairs,),
            None,
            
            # Args
            dt_step,
            pairs,
            pos,
            vel,
            mass,
            radius,
            toi, 
            vel_table,
        )
