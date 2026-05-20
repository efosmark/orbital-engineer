import math
import cairo
from orbitalengineer.ui.canvas import renderer

MIN_RADIUS = 0.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

class SelectionRenderer(renderer.Renderer):


    def draw(self, cr:cairo.Context, width:int, height:int):
        cr.set_line_width(0.5)
        
        if self.view.props.dragging_particle is not None:
            return
        
        if self.view.drag_start and self.view.drag_end:
            cr.set_source_rgba(1,1,1,1)
            
            x1,y1 = self.view.drag_start
            x2,y2 = self.view.drag_end
            cr.set_dash([2.0/self.camera.zoom, 2.0/self.camera.zoom])
            cr.rectangle(x1, y1, x2-x1, y2-y1)
            cr.set_line_width(1/self.camera.zoom)
            cr.stroke()

        if self.view.selected_particles is None:
            return
        
        cr.set_dash([])
        for idx in self.view.selected_particles:
            b = self.orbital.get_particle(idx)
            radius = b.get_radius()
            
            cr.set_source_rgba(1,1,1,0.7)
            x,y = b.get_xy()
            cr.arc(x, y, radius + 1, 0, 2*math.pi)
            cr.set_line_width(3/self.camera.zoom)
            cr.stroke()        
                