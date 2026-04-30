import pyopencl as cl
import numpy as np
from orbitalengineer.engine.cl.old.kernelizer import KernelBuilder
mf = cl.mem_flags


kernel_src = """
#include "kernel/complex.clh"
#include "kernel/pair.clh"
#include "kernel/force.clh"
#include "kernel/toi.clh"
#include "kernel/scalar_impulse_magnitude.clh"


inline float compute_time_of_impact(
    const uint    i,
    const uint    j,
    const cfloat* position,
    const cfloat* velocity,
    const float*  mass,
    const float*  radius
) {

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
    bool is_touching = edge_dist <= EPS_DIST;

    float toi = is_approaching ? q.s0 : INFINITY;
    return is_approaching && is_touching ? 0 : toi;
}


inline cfloat kick_pairwise(
               const uint    i,
               const uint    j,
               const float   dt,
               const cfloat* position,
               const cfloat* velocity,
               const float*  mass,
               const float*  radius
) {
    float dist = fast_distance(position[j], position[i]);
    cfloat dP = position[j] - position[i]; // Relative position
    float inv_mass_i = 1.0 / mass[i];
    float R = radius[j] + radius[i];
    float edge_dist = dist - R;
    

    // Apply Netwon's theory of universal gravitation
    cfloat f = dist > 0 ? force(i, j, dP, dist, mass) : 0;
    cfloat result = f * inv_mass_i * dt;
    
    
    if (edge_dist <= 0) { // BOUNCE
    
        cfloat r_norm = normalize(dP);
        float impulse = scalar_impulse_magnitude(i, j, r_norm, velocity, mass);
        result += r_norm * (impulse * inv_mass_i);
    }
    
    return result;
}

inline cfloat kick(
               const uint    N,
               const uint    node,
               const uint    Lx,
               const uint    lane,
                     float   dt,
    __global         cfloat* position,
    __global         cfloat* velocity,
    __global         float*  mass,
    __global         float*  radius
) {
    
    // grid-stride accumulator
    float2 vel_accum = (float2)(0.0f, 0.0f);
    for (uint j = lane; j < N; j += Lx) {
        cfloat v = kick_pairwise(node, j, dt, position, velocity, mass, radius);
        vel_accum += v;
    }
    
    float real = vel_accum.x;
    float imag = vel_accum.y;

    float sx = work_group_reduce_add(real);
    float sy = work_group_reduce_add(imag);

    return (float2)(sx, sy);
}

__kernel void substep(
             const uint    N,
                   float   dt,
    __global       cfloat* position,
    __global       cfloat* velocity,
    __global       float*  mass,
    __global       float*  radius
) {
    uint node = get_group_id(0);
    //if (node >= N) return;
    
    uint lane = get_local_id(0);
    uint Lx   = get_local_size(0);

    barrier(CLK_GLOBAL_MEM_FENCE);

    float2 vel_accum = (float2)(0.0f, 0.0f);
    for (uint j = lane; j < N; j += Lx) {
        cfloat v = kick_pairwise(node, j, dt, position, velocity, mass, radius);
        vel_accum += v;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    float real = vel_accum.x;
    float imag = vel_accum.y;

    float sx = work_group_reduce_add(real);
    float sy = work_group_reduce_add(imag);

    barrier(CLK_GLOBAL_MEM_FENCE);

    //cfloat v = kick(N, node, Lx, dt, lane, position, velocity, mass, radius);
    if (lane == 0) {
        velocity[node] += (float2)(sx, sy);;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE);

    if (lane == 0){
        position[node] += velocity[node] * dt;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE);
    
    
    vel_accum = (float2)(0.0f, 0.0f);
    for (uint j = lane; j < N; j += Lx) {
        cfloat v = kick_pairwise(node, j, dt, position, velocity, mass, radius);
        vel_accum += v;
    }
    
    real = vel_accum.x;
    imag = vel_accum.y;

    sx = work_group_reduce_add(real);
    sy = work_group_reduce_add(imag);
    
    //v = kick(N, node, Lx, dt, lane, position, velocity, mass, radius);
    if (lane == 0) {
        velocity[node] += (float2)(sx, sy);
    }
    
    //barrier(CLK_GLOBAL_MEM_FENCE);
}
"""

class SubStep(KernelBuilder):
    kernel_name = "substep"
    kernel_src = kernel_src

    def __call__(
        self,
        N: int, 
        dt_value: float,
        pos: cl.Buffer,
        vel: cl.Buffer,
        mass: cl.Buffer,
        radius: cl.Buffer,
        *,
        local_size:int=64,
        metric_alias:str|None=None
    ):
        Lx = min(N, int(local_size))
        return self.kernel(
            self.queue,
            (N * Lx, ), # global work size
            (Lx, ),     # local work size
            
            # Args
            np.uint32(N),
            dt_value,
            pos,
            vel,
            mass,
            radius,
            
            metric_alias=metric_alias
        )


if __name__ == "__main__":
    ...