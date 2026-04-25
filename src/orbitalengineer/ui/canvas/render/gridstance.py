import math
import cairo
from orbitalengineer.ui.gtk4 import Gtk, Graphene, Gdk
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.fmt import mag_format

LABELS_RGB = (0.2, 0.2, 0.2)
LINES_RGB =  (0.08, 0.08, 0.1)

LABEL_HL_RGB = (0.7, 0.7, 0.7)
LINE_HL_RGB = (0.3, 0.3, 0.3)

LABEL_PADDING = 5
PRIMARY_SPACING = 0.5
SECONDARY_SPACING = 0.1

def power_of(x, n):
    try:
        exp = int(math.floor(math.log10(abs(x)) / float(n)) * n)
    except:
        exp = 1
    mant = x / (10 ** exp)
    return mant, exp

def find_offset(a1, a2, spacing, scale=1):
    # Find the earliest spot where the axis lines up with the spacing
    delta = a2 - a1
    p1a = a1 - (a1 % spacing)
    offset_p = delta - (a2 - p1a)
    offset_p = offset_p / scale
    return offset_p


###### X AXIS #############

def draw_x_line(cr, x1, x_offset, height, scale, fade, show_label):
    line = x_offset / scale
    cr.set_source_rgba(*LINES_RGB, fade)
    cr.move_to(line, 0)
    cr.line_to(line, height)
    cr.stroke()


def x_lines(cr, height, x1, x2, spacing, scale, fade=1.0, show_label=True):
    delta = x2 - x1
    x_offset = find_offset(x1, x2, delta, spacing)
    while x_offset < delta:
        draw_x_line(cr, x1, x_offset, height, scale, fade, show_label)
        x_offset += spacing


###### Y AXIS #############

def draw_y_line(cr, y1, y_offset, width, scale, fade):
    line = y_offset / scale
    cr.set_source_rgba(*LINES_RGB, fade)
    cr.move_to(0, line)
    cr.line_to(width, line)
    cr.stroke()

def y_lines(cr, width, y1, y2, spacing, scale, fade=1.0, show_label=True):
    delta = y2 - y1
    y_offset = find_offset(y1, y2, delta, spacing)
    while y_offset < delta:
        draw_y_line(cr, y1, y_offset, width, scale, fade)
        y_offset += spacing


class GridstanceRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        
        if not self.data.secondary_body:
            return
        
        p = self.orbital.get_particle(self.data.secondary_body)
        sx, sy = p.get_xy()
        radius = int(p.get_radius())
       
        # Secondary (smaller) grid
        cr.set_source_rgba(1, 1, 1, 0.2)
        
        #wx, wy = self.camera.world_to_screen(sx, sy, width, height)
        
        cr.set_line_width(1/self.camera.zoom)
        
        dist = radius
        
        for i in range(10):
            cr.move_to(sx+dist, sy-dist)
            cr.line_to(sx+dist, sy+dist)
            
            

            cr.stroke()
            dist = 2**i
        