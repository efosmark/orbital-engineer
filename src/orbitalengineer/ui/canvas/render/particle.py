import math
import cairo
from orbitalengineer.ui.canvas import renderer

MIN_DISPLAY_RADIUS = 1.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

class ParticleRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        
        for b in self.orbital:
            try:
                radius = b.get_radius()
            except AttributeError:
                raise SystemExit
            if b.get_mass() == 0:
               continue
            
            x,y = b.get_xy()
            
            radius = max(radius, MIN_DISPLAY_RADIUS/self.camera.zoom)
            
            cr.set_source_rgba(*self.view.props.particle_colors[b.idx])
            cr.arc(x, y, radius, 0, 2*math.pi)
            cr.fill()
