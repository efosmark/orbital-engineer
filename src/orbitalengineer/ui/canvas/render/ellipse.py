from typing import cast
import cairo
from orbitalengineer.engine.particle_cl import ParticleCL
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.canvas import odt

class EllipseRenderer(renderer.Renderer):

    def draw_ellipse_for_body(self, cr:cairo.Context, body_id:int, faded:bool=False):
        secondary = cast(ParticleCL, self.orbital.get_particle(body_id))
                        
        # TODO: Don't run this in the rendering code. Only run once per tick.
        o = secondary.get_orbit_info()
        if not o: return
        
        self.odt = odt.OrbitDrawingTool()
        
        show_detailed_view = (not faded) and self.app.show_debug_info
        self.odt.draw_orbit(
            cr,
            self.camera.zoom,
            o,
            #show_semimajor_axis=show_detailed_view,
            #show_anomaly=show_detailed_view,
            apsis_radius=secondary.get_radius() if show_detailed_view else None
        )

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.app.show_orbital_ellipse: return
        if self.app.secondary_body is not None:
            self.draw_ellipse_for_body(cr, self.app.secondary_body)
        
        b = self.app.hovered_over_particle
        if b is not None and b.idx != self.app.secondary_body:
            self.draw_ellipse_for_body(cr, b.idx, True)