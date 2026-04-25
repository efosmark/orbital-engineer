import numpy as np
import cairo
from orbitalengineer.ui.canvas import renderer


class HistoryRenderer(renderer.Renderer):

    def get_history(self, sb_id):
        history_idx = self.orbital.shm.history_index[sb_id]
        history:list = np.roll(self.orbital.shm.history[sb_id], -history_idx-1).tolist()
        history = [h for h in history if h != 0]
        return history

    def draw_history(self, cr:cairo.Context, sb_id):
        cr.save()
        
        r,g,b,a = self.view.particle_colors[sb_id]
        cr.set_source_rgba(r, g, b, 0.2)
        cr.set_line_width(1/self.camera.zoom)
        cr.set_line_join(cairo.LineJoin.MITER)
        last_position = None
        for position in self.get_history(sb_id):
            if last_position is None:
                last_position = position
                cr.move_to(last_position.real, last_position.imag)
                continue
            cr.move_to(last_position.real, last_position.imag)
            cr.line_to(position.real, position.imag)
            cr.stroke()
            last_position = position
        if last_position is not None:
            buffer_id = self.orbital.buffer_static
            position = self.orbital.shm.position[buffer_id, sb_id]
            cr.move_to(last_position.real, last_position.imag)
            cr.line_to(position.real, position.imag)
            cr.stroke()
        cr.restore()


    def draw(self, cr:cairo.Context, width:int, height:int):
        if self.view.show_all_history:
            for b in self.orbital:
                self.draw_history(cr, b.idx)
        elif self.view.show_focused_history and self.data.secondary_body is not None:
            self.draw_history(cr, self.data.secondary_body)