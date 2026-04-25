import cairo
from orbitalengineer.ui.canvas import renderer

class BackgroundRenderer(renderer.Renderer):
    
    def draw(self, cr:cairo.Context, width:int, height:int):
        cr.set_source_rgb(0, 0, 0.005) 
        cr.paint()