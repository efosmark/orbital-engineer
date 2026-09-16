from collections import deque
import math

import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.color import set_source_hex
from orbitalengineer.ui.fmt import format_time

X_PADDING = 10
X_MARGIN = 5

Y_PADDING = 6
Y_MARGIN = 10

FONT_SIZE = 10


class HudClockRenderer(renderer.Renderer):
    
    def initialize(self):
        self.avg_time_error = deque(maxlen=60)
        self._last_tick = self.app.engine.tick_id
    
    def record_time_error(self):
        if not self.orbital.clock.running or self.app.engine.tick_id == self._last_tick:
            return
        time_error = self.app.engine.curr_tick_at - self.app.engine.next_tick_at
        if time_error < 0:
            time_error = 0
        self.avg_time_error.append(time_error)
        self._last_tick = self.app.engine.tick_id
    
    def get_time_error(self):
        try:
            return sum(self.avg_time_error)/len(self.avg_time_error)
        except ZeroDivisionError:
            return 0.0
    
    def draw(self, cr, width:int, height:int):
        bg_color = "#8E535354" if self.app.engine.paused else "#538E8855"
        border_color = "#4F252599" if self.app.engine.paused else "#1C393699"
        
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

        speed = round(self.app.engine.clock_speed, 6)
        status = "  " if self.app.engine.paused else "▶"

        txt = f"[{status}]   T+{format_time(self.clock.time())}   ({speed:.2f}x)"

        te = cr.text_extents(txt)
        r_height = te.height + (Y_PADDING * 2)
        r_width = te.width + (X_PADDING * 2)
    
        cr.save()
        cr.translate(X_MARGIN, height - r_height - Y_MARGIN)
        cr.move_to(0, 0)

        # Background
        set_source_hex(cr, border_color)
        cr.rectangle(0, 0, r_width, r_height)
        cr.stroke()
        
        set_source_hex(cr, bg_color)
        cr.rectangle(0, 0, r_width, r_height)
        cr.fill()

        cr.move_to(X_PADDING, Y_PADDING + te.height)
        set_source_hex(cr, "#FFFFFF")
        cr.show_text(txt)

        self.record_time_error()
        time_error = self.get_time_error()

        meter_height = 5
        meter_width = 150
        meter_y_start = 0.0 - meter_height - 5
        
        set_source_hex(cr, "#6161612E")
        cr.set_hairline(True)
        cr.rectangle(0, meter_y_start, meter_width, meter_height)
        cr.fill()
        
        meter_position_x = 0
        if time_error > 0:
            time_error_steps = (time_error/self.orbital.dt_step)            
            x = time_error_steps
            log_val = math.log(x+1) / math.log(10)
            scaled = (log_val / (log_val + 1))
            meter_position_x = scaled * meter_width
        
        set_source_hex(cr, "#FFFFFF2F")
        cr.rectangle(0, meter_y_start, meter_position_x, meter_height)
        cr.fill()
        cr.restore()
