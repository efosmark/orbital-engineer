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
BG_COLOR = "#0A121A"
TEXT_COLOR = "#95B2CD"
BORDER_COLOR = "#1C3247"


class HudClockRenderer(renderer.Renderer):
    
    def draw(self, cr, width:int, height:int):
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

        txt = f"T+ {self.orbital.clock.time():5.2f}"

        te = cr.text_extents(txt)
        r_height = te.height + (Y_PADDING * 2)
        r_width = te.width + (X_PADDING * 2)
    
        cr.save()
        cr.translate(X_MARGIN, height - r_height - Y_MARGIN)
        cr.move_to(0, 0)

        # Background
        set_source_hex(cr, BORDER_COLOR)
        cr.rectangle(0, 0, r_width, r_height)
        cr.stroke()
        
        set_source_hex(cr, BG_COLOR)
        cr.rectangle(0, 0, r_width, r_height)
        cr.fill()

        cr.move_to(X_PADDING, Y_PADDING + te.height)
        set_source_hex(cr, TEXT_COLOR)
        cr.show_text(txt)

        cr.restore()
