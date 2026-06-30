import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.color import set_source_hex
from orbitalengineer.ui.fmt import mag_format

MARGIN = 0
X_PADDING = 10
Y_PADDING = 2
FONT_SIZE = 10

class WarningRenderer(renderer.Renderer):
    
    def draw(self, cr, width:int, height:int):
        step_overflow = (self.orbital.accum // self.orbital.dt_base)
        if step_overflow <= 10: return

        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(FONT_SIZE)
        
        txt = f"Lagging by {mag_format(step_overflow)} steps. Reduce speed to let it catch up."
        te = cr.text_extents(txt)
        r_height = te.height + (Y_PADDING * 2)
        r_width = te.width + (X_PADDING * 2)
    
        cr.save()
        cr.translate(width - r_width, height - r_height)
        cr.move_to(0, 0)
        
        set_source_hex(cr, "#721010FF")
        cr.rectangle(0, 0, r_width, r_height)
        cr.fill()

        cr.move_to(X_PADDING, Y_PADDING + te.height/1.25)
        set_source_hex(cr, "#FFFFFF99")
        cr.show_text(txt)

        cr.restore()