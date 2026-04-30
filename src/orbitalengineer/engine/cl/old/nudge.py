import pyopencl as cl

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
#include "kernel/complex.clh"
#include "kernel/pair.clh"
#include "kernel/toi.clh"

__kernel void nudge_overlaps(
    __constant       Pair*   pair,
    __global   const cfloat* position,
    __global   const cfloat* velocity,
    __global   const float*  mass,
    __global   const float*  radius,
    __global         float*  toi,
    __global         cfloat* pair_pos_ij,
    __global         cfloat* pair_pos_ji
) {
    uint ij = get_global_id(0);

    pair_pos_ij[ij] = 0;
    pair_pos_ji[ij] = 0;
    
    uint i = pair[ij].i;
    uint j = pair[ij].j;

    float R = radius[j] + radius[i];
    float dist = fast_distance(position[j], position[i]);
    float edge_dist = dist - R;

    if (edge_dist <= 0) {
        cfloat dP = position[j] - position[i]; // Relative position
        cfloat r_norm = normalize(dP);
        float inv_mass_i = 1.0 / mass[i];
        float inv_mass_j = 1.0 / mass[j];
        float inv_mass_sum = (inv_mass_i + inv_mass_j);

        float k = edge_dist / inv_mass_sum;
        pair_pos_ij[ij] += r_norm * (k * inv_mass_i);
        pair_pos_ji[ij] -= r_norm * (k * inv_mass_j);
        
        
        cfloat dV = velocity[j] - velocity[i];
        dP = (position[j] + pair_pos_ji[ij]) - (position[i] + pair_pos_ij[ij]);
        toi[ij] = 0;
    }
}
"""


class NudgeOverlaps(KernelBuilder):
    kernel_name = "nudge_overlaps"
    kernel_src = kernel_src

    def __call__(
        self,
        num_pairs: int,
        pairs: cl.Buffer,
        pos: cl.Buffer,
        vel: cl.Buffer,
        mass: cl.Buffer,
        radius: cl.Buffer,
        toi: cl.Buffer,
        pair_pos_ij: cl.Buffer,
        pair_pos_ji: cl.Buffer,
        
        metric_alias:str|None=None
    ):
        return self.kernel(
            self.queue,
            (num_pairs,),
            None,
            
            # Args
            pairs,
            pos, vel, mass, radius,
            toi, pair_pos_ij, pair_pos_ji,
            
            metric_alias=metric_alias
        )