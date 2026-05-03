from collections import defaultdict, deque
import numpy as np
import cairo
from orbitalengineer.ui.canvas import renderer


class HistoryRenderer(renderer.Renderer):
    history_positions = defaultdict(lambda:deque(maxlen=200))


    def draw_history(self, cr:cairo.Context, sb_id):
        cr.save()

        particle = self.orbital.get_particle(sb_id)
        r,g,b,a = self.view.particle_colors[sb_id]
        #cr.set_line_join(cairo.LineJoin.MITER)        
        
        num_positions = float(len(self.history_positions[sb_id]))
        current_position = particle.get_position()
        radius = particle.get_radius() / 2.0
        last_position = None
        for i,position in enumerate([*self.history_positions[sb_id], current_position]):
            if last_position is None:
                last_position = position
                cr.move_to(last_position.real, last_position.imag)
                continue
            
            pct = (i/num_positions) * 0.9
            
            cr.set_line_width(3.0/self.camera.zoom)
            cr.set_source_rgba(r, g, b, pct)
            cr.move_to(last_position.real, last_position.imag)
            cr.line_to(position.real, position.imag)
            cr.stroke()
            last_position = position
            
    
        #if self.orbital.tick_id % 2 == 0:
        self.history_positions[sb_id].append(current_position)

        cr.restore()


    def draw(self, cr:cairo.Context, width:int, height:int):
        if self.view.show_all_history:
            for b in self.orbital:
                self.draw_history(cr, b.idx)
        elif self.view.show_focused_history and self.data.secondary_body is not None:
            self.draw_history(cr, self.data.secondary_body)