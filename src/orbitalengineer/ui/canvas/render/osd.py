import time
from typing import cast

import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.model import OSDMessage

FONT_SIZE = 20
TEXT_COLOR = (0.85, 0.85, 0.85)
PADDING_Y = 10

class OSDRenderer(renderer.Renderer):

    def draw(self, cr, width:int, height:int):
        if len(self.view.osd_message) == 0: return
                
        now = time.monotonic()
        m = cast(OSDMessage, self.view.osd_message[0])
        if m.start is None:
            m.start = now
        
        duration = now - m.start
        remaining = m.duration - duration
        intensity = 1.0
        if remaining <= min(1.0, m.duration):
            intensity = remaining/min(1.0, m.duration)
        
        cr.save()
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

        te = cr.text_extents(m.message)        
        box_height = FONT_SIZE + (PADDING_Y * 2.0)
        
        y_start = (height * 0.75)
        y_end = y_start + box_height
        
        lg_background = cairo.LinearGradient(0, y_start, width, y_end)
        lg_background.add_color_stop_rgba(0.00, 0.0, 0.0, 0.0, 0.1 * intensity)
        lg_background.add_color_stop_rgba(0.25, 0.0, 0.1, 0.1, 1.0 * intensity)
        lg_background.add_color_stop_rgba(0.75, 0.0, 0.1, 0.1, 1.0 * intensity)
        lg_background.add_color_stop_rgba(1.00, 0.0, 0.0, 0.0, 0.1 * intensity)
                
        cr.rectangle(0, y_start, width, box_height)
        cr.set_source(lg_background)
        cr.fill()
        
    
        lg_border = cairo.LinearGradient(0, y_start, width, y_end)
        lg_border.add_color_stop_rgba(0.00, 0.0, 0.0, 0.0, 0.0)
        lg_border.add_color_stop_rgba(0.07, 1.0, 1.0, 1.0, 0.2 * intensity)
        lg_border.add_color_stop_rgba(0.75, 1.0, 1.0, 1.0, 0.9 * intensity)
        lg_border.add_color_stop_rgba(0.93, 1.0, 1.0, 1.0, 0.2 * intensity)
        lg_border.add_color_stop_rgba(1.00, 0.0, 0.0, 0.0, 0.0)
        
        cr.set_source(lg_border)
        cr.move_to(0, y_start)
        cr.line_to(width, y_start)
        cr.stroke()
        
        cr.move_to(0, y_end)
        cr.line_to(width, y_end)
        cr.stroke()
        
        text_start_x = width/2.0 - (te.width/2.0)
        text_start_y = y_start + PADDING_Y - te.y_bearing
        
        cr.move_to(text_start_x, text_start_y)
        cr.set_source_rgba(*TEXT_COLOR, intensity)
        cr.show_text(m.message)
        cr.restore()
        
        if now - m.start > m.duration:
            self.view.osd_message.remove(m)
