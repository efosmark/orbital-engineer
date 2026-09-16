from typing import cast

import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.model import OSDMessage

SLIDE_IN_TIME = 0.1
SLIDE_OUT_TIME = 0.25

Y_OFFSET_FACTOR = 0.75

FONT_FACE = ("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

FONT_SIZE = 20
PADDING_Y = 10
TEXT_COLOR = (0.85, 0.85, 0.85)
BORDER_COLOR = (0.85, 0.85, 0.85)
BG_COLOR = (0.0, 0.1, 0.1)

DESC_FONT_SIZE = 10
DESC_PADDING_Y = 10
DESC_TEXT_COLOR = (0.65, 0.65, 0.65)
DESC_BG_COLOR = (0.0, 0.0, 0.0)

class OSDRenderer(renderer.Renderer):

    def draw(self, cr, width:int, height:int):
        osd = self.app.osd
        
        m = osd.get_message()
        if m is None:
            return
                
        duration = m.now - (m.start or 0)
        
        intensity = 1.0
        remaining = m.duration - duration
        if remaining <= min(0.5, m.duration):
            intensity = remaining/min(0.5, m.duration)
        
        border_slide = 1.0
        if duration <= SLIDE_IN_TIME and m.fade_in:
            border_slide = duration/SLIDE_IN_TIME
        elif remaining <= SLIDE_OUT_TIME:
            border_slide = remaining/SLIDE_OUT_TIME
                
        cr.save()
        cr.select_font_face(*FONT_FACE)
        cr.set_font_size(FONT_SIZE)

        te = cr.text_extents(m.message)        
        box_height = FONT_SIZE + (PADDING_Y * 2.0)
        
        y_start = height * Y_OFFSET_FACTOR
        y_end = y_start + box_height
        
        lg_stop_offset = (0.4 * (1.0 - border_slide))
        
        self._draw_background(cr, y_start, width, box_height, lg_stop_offset)
        self._draw_border(cr, y_start, y_end, width, lg_stop_offset)
        
        text_start_x = width/2.0 - (te.width/2.0)
        text_start_y = y_start + PADDING_Y - te.y_bearing
        
        cr.move_to(text_start_x, text_start_y)
        cr.set_source_rgba(*TEXT_COLOR, intensity)
        cr.show_text(m.message)

        if m.description is not None:
            self._draw_description(cr, m, y_end, width, height, lg_stop_offset, intensity)

        cr.restore()        

    def _draw_background(self, cr:cairo.Context, y_start:float, width:float, height:float, lg_stop_offset:float):
        lg_background = cairo.LinearGradient(0, y_start, width, y_start + height)
        lg_background.add_color_stop_rgba(0.00, 0, 0, 0, 0.0)
        lg_background.add_color_stop_rgba(0.10 + lg_stop_offset, *BG_COLOR, 0.3)
        lg_background.add_color_stop_rgba(0.50, *BG_COLOR, 1.0)
        lg_background.add_color_stop_rgba(0.90 - lg_stop_offset, *BG_COLOR, 0.3)
        lg_background.add_color_stop_rgba(1.00, 0, 0, 0, 0.0)

        cr.rectangle(0, y_start, width, height)
        cr.set_source(lg_background)
        cr.fill()
    
    def _draw_border(self, cr:cairo.Context, y_start, y_end, width, lg_stop_offset):
        lg_border = cairo.LinearGradient(0, y_start, width, y_end)
        lg_border.add_color_stop_rgba(0.01, 0.0, 0.0, 0.0, 0.0)
        lg_border.add_color_stop_rgba(0.05 + lg_stop_offset, *BORDER_COLOR, 0.0)
        lg_border.add_color_stop_rgba(0.50, *BORDER_COLOR, 1.0)
        lg_border.add_color_stop_rgba(0.95 - lg_stop_offset, *BORDER_COLOR, 0.0)
        lg_border.add_color_stop_rgba(0.99, 0.0, 0.0, 0.0, 0.0)
        
        cr.set_source(lg_border)
        cr.set_line_width(1.0)
        cr.set_hairline(True)
        cr.move_to(0, y_start)
        cr.line_to(width, y_start)
        cr.stroke()
        
        cr.move_to(0, y_end)
        cr.line_to(width, y_end)
        cr.stroke()

    def _draw_description(self, cr:cairo.Context, m, y_end:float, width:float, height:float, lg_stop_offset:float, intensity):
        cr.set_font_size(DESC_FONT_SIZE)
        te = cr.text_extents(m.description)
        text_start_x = width/2.0 - (te.width/2.0)
        text_start_y = y_end + DESC_PADDING_Y - te.y_bearing
        
        lg_border = cairo.LinearGradient(0, y_end, width, height)
        lg_border.add_color_stop_rgba(0.05 + lg_stop_offset, *DESC_BG_COLOR, 0.0)
        lg_border.add_color_stop_rgba(0.50, *DESC_BG_COLOR, 1.0)
        lg_border.add_color_stop_rgba(0.95 - lg_stop_offset, *DESC_BG_COLOR, 0.00)
        
        cr.set_source(lg_border)
        cr.rectangle(0, y_end + 1, width, (DESC_PADDING_Y * 2.0) - te.y_bearing)
        cr.fill()
                    
        cr.move_to(text_start_x, text_start_y)
        cr.set_source_rgba(*DESC_TEXT_COLOR, intensity)
        cr.show_text(m.description)