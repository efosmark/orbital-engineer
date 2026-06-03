import math
import statistics
from dataclasses import dataclass
from typing import cast
from pathlib import Path
import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.color import set_source_hex
from orbitalengineer.ui.gtk4 import Gdk

X_PADDING = 15
X_MARGIN = 10

Y_PADDING = 10
Y_MARGIN = 10

FONT_SIZE = 14


class HudClockRenderer(renderer.Renderer):
    
    def draw(self, cr, width:int, height:int):
        bg_color = "#8E535354" if self.view.paused else "#538E8855"
        border_color = "#4F252599" if self.view.paused else "#1C393699"
        
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

        txt = f"T+ {self.clock.time():5.2f}"

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




        cr.set_font_size(10)
        txt = "PAUSED" if self.view.paused else "RUNNING"
        te = cr.text_extents(txt)

        status_padding = te.height#/2.0

        cr.save()
        cr.translate(0, -te.height - (status_padding * 2) - Y_MARGIN)
        

        set_source_hex(cr, border_color)
        cr.rectangle(0, 0, r_width, te.height + (status_padding * 2))
        cr.stroke()
        
        set_source_hex(cr, bg_color)
        cr.rectangle(0, 0, r_width, te.height + (status_padding * 2))
        cr.fill()
        
        
        
        set_source_hex(cr, "#FFFFFF")
        cr.move_to(X_PADDING, te.height + status_padding)
        cr.show_text(txt)

        cr.restore()
        cr.restore()
