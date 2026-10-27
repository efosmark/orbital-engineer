from typing import cast
import cairo
from gui.canvas import renderer
from gui.canvas import odt

from orbitalengineer import body
from orbitalengineer.twobody import orbit

class EllipseRenderer(renderer.Renderer):

    def draw_ellipse_for_body(self, cr:cairo.Context, body_id:int, faded:bool=False):
        secondary = cast(body.Body, self.app.engine.get_particle(body_id))
        
        focus = orbit.get_focus(body_id, self.app.engine.N, self.app.engine.force, self.app.engine)
        if focus is None: return
        
        primary, influence = focus
        
        # TODO: Don't run this in the rendering code. Only run once per tick.
        o = orbit.get_orbit_info(body_id, primary, self.app.engine)
        if not o: return
        
        self.odt = odt.OrbitDrawingTool()
        
        show_detailed_view = (not faded) and self.app.show_debug_info
        self.odt.draw_orbit(
            cr,
            self.camera.zoom,
            o,
            #show_semimajor_axis=show_detailed_view,
            #show_anomaly=show_detailed_view,
            apsis_radius=secondary.radius if show_detailed_view else None
        )

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.app.show_orbital_ellipse: return
        if self.app.secondary_body is not None:
            self.draw_ellipse_for_body(cr, self.app.secondary_body)
        
        b = self.app.hovered_over_particle
        if b is not None and b.index != self.app.secondary_body:
            self.draw_ellipse_for_body(cr, b.index, True)