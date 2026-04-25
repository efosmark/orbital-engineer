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

def find_offset(a1, a2, delta_a, spacing, scale=1):
    # Find the earliest spot where the axis lines up with the spacing
    delta = a2 - a1
    p1a = a1 - (a1 % spacing)
    offset_p = delta - (a2 - p1a)
    offset_p = offset_p / scale
    return offset_p

def get_label(value):
    if abs(value)*1.05 >= 1e3:
        return mag_format(value, sig=1)
    elif abs(value) < 1e-6:
        return "0"
    elif abs(value) < 1:
        return f"{value:.2f}"
    else:
        return f"{value:.0f}"


###### X AXIS #############


def draw_x_label(cr:cairo.Context, x1, x_offset, height, scale, highlight_label):
    line = x_offset / scale

    if highlight_label:
        cr.set_source_rgb(*LINE_HL_RGB)
        cr.set_dash([2, 2])
        cr.move_to(line, height-60)
        cr.line_to(line, height)
        cr.stroke()
        cr.set_source_rgb(*LABEL_HL_RGB)
    else:
       cr.set_source_rgb(*LABELS_RGB)

    label = get_label(x1 + x_offset)
    
    # Rotate the label
    cr.save()
    cr.translate(line, height)
    cr.move_to(-LABEL_PADDING, -LABEL_PADDING)
    cr.rotate(3.14/-2)
    cr.show_text(label)
    cr.restore()


def draw_x_line(cr, x1, x_offset, height, scale, fade, show_label):
    line = x_offset / scale
    cr.set_source_rgba(*LINES_RGB, fade)
    cr.move_to(line, 0)
    cr.line_to(line, height)
    cr.stroke()
    if show_label:
        draw_x_label(cr, x1, x_offset, height, scale, False)


def x_lines(cr, height, x1, x2, spacing, scale, fade=1.0, show_label=True):
    delta = x2 - x1
    x_offset = find_offset(x1, x2, delta, spacing)
    while x_offset < delta:
        draw_x_line(cr, x1, x_offset, height, scale, fade, show_label)
        x_offset += spacing


###### Y AXIS #############


def draw_y_label(cr, y1, y_offset, width, scale, highlight_label):
    line = y_offset / scale
    if highlight_label:
        cr.set_source_rgb(*LINE_HL_RGB)
        cr.set_dash([2, 2])
        cr.move_to(width-60, line)
        cr.line_to(width, line)
        cr.stroke()
        cr.set_source_rgb(*LABEL_HL_RGB)
    else:
       cr.set_source_rgb(*LABELS_RGB)
        
    label = get_label(y1 + y_offset)
    te = cr.text_extents(label)
    cr.move_to(
        width - te.width - LABEL_PADDING,
        line - te.y_advance - LABEL_PADDING
    )
    cr.show_text(label)


def draw_y_line(cr, y1, y_offset, width, scale, fade, show_label):
    line = y_offset / scale
    cr.set_source_rgba(*LINES_RGB, fade)
    cr.move_to(0, line)
    cr.line_to(width, line)
    cr.stroke()
    if show_label:
        draw_y_label(cr, y1, y_offset, width, scale, False)


def y_lines(cr, width, y1, y2, spacing, scale, fade=1.0, show_label=True):
    delta = y2 - y1
    y_offset = find_offset(y1, y2, delta, spacing)
    while y_offset < delta:
        draw_y_line(cr, y1, y_offset, width, scale, fade, show_label)
        y_offset += spacing


class GridRenderer(renderer.Renderer):
    
    # def get_cache_key(self, width: int, height: int):
    #     x1, y1 = self.camera.screen_to_world(0, 0, width, height)
        # return (*super().get_cache_key(width, height), int(x1), int(y1))

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_grid:
            return
        
        cr.select_font_face(self.view.font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(self.view.font_size)
        
        cr.set_hairline(True)
        cr.set_antialias(cairo.ANTIALIAS_FAST)
        
        x1, y1 = self.camera.screen_to_world(0, 0, width, height)
        x2, y2 = self.camera.screen_to_world(width, height, width, height)        
        
        dx = x2 - x1
        scale = (x2 - x1) / width

        # 1. Find the Pow10 for dx
        base, exp = power_of(dx, 1)
        
        # 2. Set the spacing as multiples of 10^x
        spacing = (10 ** exp) * PRIMARY_SPACING
        
        # Normalize the fade factor
        # The `base` is a value spanning 1-9, so this will make it a value 0.0 - 1.0
        fade = 1 - ((base-1) / 9)

        # Secondary (smaller) grid
        cr.set_source_rgb(*LINES_RGB)
        
        secondary_spacing = spacing * SECONDARY_SPACING
        x_lines(cr, height, x1, x2, secondary_spacing, scale, fade, show_label=False)
        y_lines(cr, width, y1, y2, secondary_spacing, scale, fade, show_label=False)
        
        # Primary, labeled, grid
        x_lines(cr, height, x1, x2, spacing, scale)
        y_lines(cr, width, y1, y2, spacing, scale)
        
        if self.data.secondary_body:
            b = self.orbital.get_particle(self.data.secondary_body)
            pos = b.get_position()
        
            draw_x_label(cr, x1, pos.real-x1, height, scale, True)
            draw_y_label(cr, y1, pos.imag-y1, width, scale, True)