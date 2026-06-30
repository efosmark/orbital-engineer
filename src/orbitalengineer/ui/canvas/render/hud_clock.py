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
    
    def draw(self, cr, width:int, height:int):
        bg_color = "#8E535354" if self.view.paused else "#538E8855"
        border_color = "#4F252599" if self.view.paused else "#1C393699"
        
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

        speed = round(self.view.speed, 6)
        status = "  " if self.view.paused else "▶"

        txt = f"[{status}]   T+{format_time(self.clock.time())}   ({speed}x)"

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

        cr.restore()
