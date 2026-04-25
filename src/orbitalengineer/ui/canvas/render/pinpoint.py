import math, time
import cairo
from orbitalengineer.ui.canvas import renderer


class PinpointRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        if len(self.view.pinpoint) == 0:
            return
                
        now = time.monotonic()
        existing_pinpoint = []
        for p in self.view.pinpoint:
            if 0 < p.until < now:
                continue
            
            total_duration = p.until - p.start
            duration = now - p.start
            intensity = (1 - (duration/total_duration))
            
            radius = (p.radius * intensity) / self.camera.zoom
            
            cr.set_source_rgba(1,1,1, intensity * 0.99)
            cr.arc(p.position.real, p.position.imag, radius, 0, 2*math.pi)
            cr.fill()
            existing_pinpoint.append(p)
        
        self.view.pinpoint = existing_pinpoint