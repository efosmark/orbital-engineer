from collections import defaultdict, deque
import cairo
from gui.canvas import renderer
from gui.ui_config import NUM_HISTORY_POSITIONS

class HistoryRenderer(renderer.Renderer):
    history_positions = defaultdict(lambda:deque(maxlen=NUM_HISTORY_POSITIONS))

    def draw_history(self, cr:cairo.Context, sb_id):
        cr.save()

        particle = self.app.engine.get_particle(sb_id)
        r,g,b,a = self.app.particle_colors[sb_id]
        
        num_positions = float(len(self.history_positions[sb_id]))
        current_position = particle.position
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
        
        self.history_positions[sb_id].append(current_position)
        cr.restore()

    def draw(self, cr:cairo.Context, width:int, height:int):
        if self.app.show_focused_history and self.app.secondary_body is not None:
            self.draw_history(cr, self.app.secondary_body)