import math
from typing import cast
import cairo
from orbitalengineer.engine import config
from orbitalengineer.engine.cl.orbitalcl import SimController_CL
from orbitalengineer.ui.canvas import renderer

import numpy as np
import pyopencl as cl

MIN_RADIUS = 0.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

class HillRadiusRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        
        orbitalcl = cast(SimController_CL, self.orbital)
        cl.enqueue_copy(orbitalcl.q, orbitalcl._hill_radius._minimum_hill_radius, orbitalcl._hill_radius._minimum_hill_radius_cl)
        min_hill_radius = orbitalcl._hill_radius._minimum_hill_radius
        
        for b in self.orbital:
            if b.get_mass() == 0: continue
                        
            color_r, color_g, color_b, _ = self.view.particle_colors.get(b.idx, DEFAULT_PARTICLE_COLOR)
            cr.set_source_rgba(color_r, color_g, color_b, 0.1)

            x,y = b.get_xy()
            cr.arc(x, y, float(min_hill_radius[b.idx]), 0, 2*math.pi)
            cr.fill()

            # cr.set_line_width(max_travel_dist)
            # cr.stroke()
            
            # if self.view.secondary_body == b.idx:
                # cr.set_source_rgba(1, 1, 1, 0.4)
                # cr.set_line_width(2)
                
                # times = cast(np.typing.NDArray, b.get_all_impact_times())
                
                # for idx, toi in enumerate(times):
                    
                #     if 0 < toi < self.orbital.dt_base:
                #         b2 = self.orbital.get_particle(idx)
                        
                #         #b2_max_travel_dist = abs(b.get_velocity()) * self.orbital.dt_base
                        
                #         #dist_c2c = abs(b2.get_position() - b.get_position()) 
                #         #dist_e2e = dist_c2c - b.get_radius() - b2.get_radius()
                #         #if dist_e2e > max_travel_dist + b2_max_travel_dist:
                #         #    print(f"[{b.idx}, {b2.idx}] out of range ({dist_e2e=:.2f}, b1_max={max_travel_dist:.2f}, b2_max={b2_max_travel_dist:.2f}, t_impact={toi:.6f})")
                #         cr.move_to(x, y)
                #         cr.line_to(*b2.get_xy())
                #         cr.stroke()