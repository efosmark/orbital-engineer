import pyopencl as cl

from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
#include "kernel/complex.clh"
#include "kernel/pair.clh"
#include "kernel/toi.clh"

__kernel void compute_impact(
    __constant       Pair*   pair,
    __global   const cfloat* position,
    __global   const cfloat* velocity,
    __global   const float*  mass,
    __global   const float*  radius,
    __global         float*  toi
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
    uint is_approaching = D > 0 && q.s0 > 0;
    bool is_touching = edge_dist <= 0;

    toi[ij] = is_approaching ? q.s0 : INFINITY;
}
"""


class ComputeImpact(KernelBuilder):
    kernel_name = "compute_impact"
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
        
        metric_alias:str|None=None
    ):
        return self.kernel(
            self.queue,
            (num_pairs,),
            None,
            
            # Args
            pairs,
            pos,
            vel,   
            mass,
            radius,
            toi,
            
            metric_alias=metric_alias
        )