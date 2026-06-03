import numpy as np
import cairo

from orbitalengineer.engine.orbitalnp.integrator.conf import GRAV_DEFAULT
from orbitalengineer.engine.orbitalnp.integrator.kick import accel
from orbitalengineer.ui.canvas import renderer

DASHES = [2, 2]

# Minimum normalized force value
# Values below this will be omitted from the display
MINIMUM_FORCE_VALUE = 0.1


class ForceVectorRenderer(renderer.Renderer):

    def get_dash_offset(self, normalized_force:float) -> float:
        dash_speed = max(0.5, normalized_force * 2.0)
        return -((self.orbital.tick_id*dash_speed) % sum(DASHES))/self.camera.zoom

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_force_vectors or self.view.secondary_body is None:
            return
        
        sb = self.orbital.get_particle(self.view.secondary_body)
        sb_pos = sb.get_position()
        
        forces = []
        for b in self.orbital:
            pos = b.get_position()
            f = abs(accel(b.get_mass(), b.get_mass(), (sb_pos - pos), GRAV_DEFAULT))
            forces.append((b.idx, pos, f))
        forces_sorted = sorted(forces, key=lambda f:f[2])[-5:]
        
        cr.set_line_cap(cairo.LineCap.ROUND)
        for b_id, b_pos, b_force in forces_sorted:
            
            # com = center_of_mass_position(
            #     np.array([b_id, sb.idx], dtype=np.int64),
            #     self.orbital.shm.mass[self.orbital.buffer_static],
            #     self.orbital.shm.position[self.orbital.buffer_static]
            # )

            cr.set_dash(
                [d/self.camera.zoom for d in DASHES],
                self.get_dash_offset(b_force)
            )
            cr.set_source_rgba(0.6, 0.6, 0.9, 0.3)
            cr.set_line_width(1/self.camera.zoom)
            
            cr.move_to(sb_pos.real, sb_pos.imag)
            #cr.line_to(com.real, com.imag)
            cr.line_to(b_pos.real, b_pos.imag)
            cr.stroke()
            
            #cr.move_to(b_pos.real, b_pos.imag)
            #cr.line_to(com.real, com.imag)
            #cr.stroke()