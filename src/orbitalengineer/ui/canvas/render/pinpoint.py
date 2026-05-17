import math
import cairo
from orbitalengineer.ui.canvas import renderer


class PinpointRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        if len(self.view.pinpoint) == 0:
            return
                
        now = self.orbital.clock.time()
        existing_pinpoint = []
        for p in self.view.pinpoint:
            if 0 < p.until < now:
                continue
            
            total_duration = p.until - p.start
            duration = now - p.start
            intensity = (1 - (duration/total_duration)) * 0.9
            radius = (p.radius * intensity) / self.camera.zoom
            
            cr.set_source_rgba(*p.color, intensity) # type: ignore
            cr.arc(p.position.real, p.position.imag, radius, 0, 2*math.pi)
            cr.fill()
            existing_pinpoint.append(p)
        
        self.view.pinpoint = existing_pinpoint