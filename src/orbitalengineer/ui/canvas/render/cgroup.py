import math
import cairo
from orbitalengineer.helpers import random_color
from orbitalengineer.ui.canvas import renderer

class CGroupRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not hasattr(self, '_colors'):
            self._colors = dict(self.view.props.particle_colors)
        
        for b in self.orbital:
            cgroup = self.orbital.cgroup[b.idx]
            if cgroup not in self._colors:
                self._colors[cgroup] = random_color()
            self.view.props.particle_colors[b.idx] = self._colors[cgroup]


class CGroupConnectionRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        cr.set_source_rgba(1, 1, 1, 0.6)
        cr.set_line_width(3.0/self.camera.zoom)
        
        for b in self.orbital:
            cgroup = int(self.orbital.cgroup[b.idx])
            if cgroup != b.idx:
                c = self.orbital.get_particle(cgroup)
                
                cr.move_to(*c.get_xy())
                cr.line_to(*b.get_xy())
                cr.stroke()