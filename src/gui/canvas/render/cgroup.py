import cairo
from orbitalengineer.helpers import random_color
from gui.canvas import renderer

class CGroupRenderer(renderer.Renderer):
    def draw(self, cr:cairo.Context, width:int, height:int):
        if not hasattr(self, '_colors'):
            self._colors = dict(self.app.particle_colors)
        for b in self.app.engine:
            cgroup = self.app.engine.cgroup[b.index]
            if cgroup not in self._colors:
                self._colors[cgroup] = random_color()
            self.app.particle_colors[b.index] = self._colors[cgroup]

class CGroupConnectionRenderer(renderer.Renderer):
    def draw(self, cr:cairo.Context, width:int, height:int):
        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.set_line_width(10.0/self.camera.zoom)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        for b in self.app.engine:
            cgroup = int(self.app.engine.cgroup[b.index])
            if cgroup != b.index:
                c = self.app.engine.get_particle(cgroup)
                cr.move_to(c.position.real, c.position.imag)
                cr.line_to(b.position.real, b.position.imag)
                cr.stroke()