import math
import cairo
from gui.canvas import renderer

MIN_DISPLAY_RADIUS = 1.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

class ParticleRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        for b in self.orbital:
            try:
                body_id = b.index
                radius = self.app.engine.radius[body_id]
                position = self.app.engine.position[body_id]
                if self.app.engine.mass[body_id] == 0: continue
            except AttributeError:  raise SystemExit
            except TypeError:       return
            
            x, y = float(position.real), float(position.imag)
            radius = max(radius, MIN_DISPLAY_RADIUS/self.camera.zoom)
            
            color = self.app.props.particle_colors.get(body_id, (1, 1, 1, 1))
            cr.set_source_rgba(*color)
            cr.arc(x, y, radius, 0, 2*math.pi)
            cr.fill()
