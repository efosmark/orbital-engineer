import time
import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.model.ledger import DISPLAY_DURATION

X_PADDING = 10
Y_PADDING = 55
FONT_SIZE = 10
TEXT_COLOR = (0.15, 0.15, 0.15)

class EventLedgerRenderer(renderer.Renderer):
    
    def draw(self, cr, width:int, height:int):
        if not self.app.show_debug_info: return

        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(FONT_SIZE)        
    
        cr.save()
        cr.translate(X_PADDING, height - Y_PADDING)
        cr.move_to(0, 0)
        
        for i, le in enumerate(reversed(self.app.ledger.get_entries())):
            cr.set_source_rgba(*TEXT_COLOR, 1.0 - ((time.monotonic() - le.start)/DISPLAY_DURATION))
            cr.move_to(0, (FONT_SIZE + 2) * -i)
            cr.show_text(str(le))
            if i > 10: break

        cr.restore()