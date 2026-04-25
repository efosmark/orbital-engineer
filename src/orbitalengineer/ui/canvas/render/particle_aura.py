import math
from typing import cast
import cairo
from orbitalengineer.engine import config
from orbitalengineer.ui.canvas import renderer

import numpy as np

MIN_RADIUS = 0.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

class ParticleAuraRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        cr.set_line_width(0.5)
        for b in self.orbital:
            radius = b.get_radius()
            x,y = b.get_xy()
            
            radius = max(radius, 0.5/self.camera.zoom)
            
            if b.get_min_toi() <= config.EPS_TIME:
                cr.set_source_rgba(1, 0.3, 0.3, 0.2)
            
            elif b.get_min_toi() == np.inf:
                cr.set_source_rgba(0.3, 0.6, 1, 0.2)
            
            else:
                cr.set_source_rgba(1, 1, 1, 0.01)
            
            max_travel_dist = abs(b.get_velocity()) * self.orbital.dt_base

            cr.arc(x, y, radius + (max_travel_dist/2.0), 0, 2*math.pi)
            cr.set_line_width(max_travel_dist)
            cr.stroke()
            
            if self.data.secondary_body == b.idx:
                cr.set_source_rgba(1, 1, 1, 0.4)
                cr.set_line_width(2)
                
                times = cast(np.typing.NDArray, b.get_all_impact_times())
                
                for idx, toi in enumerate(times):
                    
                    if 0 < toi < self.orbital.dt_base:
                        b2 = self.orbital.get_particle(idx)
                        
                        #b2_max_travel_dist = abs(b.get_velocity()) * self.orbital.dt_base
                        
                        #dist_c2c = abs(b2.get_position() - b.get_position()) 
                        #dist_e2e = dist_c2c - b.get_radius() - b2.get_radius()
                        #if dist_e2e > max_travel_dist + b2_max_travel_dist:
                        #    print(f"[{b.idx}, {b2.idx}] out of range ({dist_e2e=:.2f}, b1_max={max_travel_dist:.2f}, b2_max={b2_max_travel_dist:.2f}, t_impact={toi:.6f})")
                        cr.move_to(x, y)
                        cr.line_to(*b2.get_xy())
                        cr.stroke()