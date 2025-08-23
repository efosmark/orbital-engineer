import math
import cairo
from orbitalengineer.ui.fmt import mag_format

LABELS_RGB = (0.2, 0.2,  0.2)
LINES_RGB =  (0.04, 0.06, 0.04)

LABEL_PADDING = 5
PRIMARY_SPACING = 0.25
SECONDARY_SPACING = 0.1

def power_of(x, n):
    exp = int(math.floor(math.log10(abs(x)) / float(n)) * n)
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
    else:
        return f"{value:.1f}"

def x_lines(ctx, height, x1, x2, spacing, scale, fade=1.0, show_label=True):
    delta = x2 - x1
    x_offset = find_offset(x1, x2, delta, spacing)
    while x_offset < delta:
        line = x_offset / scale
        ctx.set_source_rgb(*[ch * fade for ch in LINES_RGB])
        ctx.move_to(line, 0)
        ctx.line_to(line, height)
        ctx.stroke()
        
        if show_label:
            label = get_label(x1 + x_offset)
            
            # Rotate the label
            ctx.save()
            ctx.translate(line, height)
            ctx.move_to(-LABEL_PADDING, -LABEL_PADDING)
            ctx.rotate(3.14/-2)
            ctx.set_source_rgb(*LABELS_RGB)
            ctx.show_text(label)
            ctx.restore()

        x_offset += spacing

def y_lines(ctx, width, y1, y2, spacing, scale, fade=1.0, show_label=True):
    delta = y2 - y1
    y_offset = find_offset(y1, y2, delta, spacing)
    while y_offset < delta:
        line = y_offset / scale
        ctx.set_source_rgb(*[ch * fade for ch in LINES_RGB])
        ctx.move_to(0, line)
        ctx.line_to(width, line)
        ctx.stroke()
        
        if show_label:
            label = get_label(y1 + y_offset)
            
            te = ctx.text_extents(label)
            ctx.move_to(
                width - te.width - LABEL_PADDING,
                line - te.y_advance - LABEL_PADDING
            )
            ctx.set_source_rgb(*LABELS_RGB)
            ctx.show_text(label)

        y_offset += spacing


def create_grid_surface(x1:float, y1:float, x2:float, y2:float, width:int, height:int):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_hairline(True)
    
    dx = x2 - x1
    scale = (x2 - x1) / width

    # 1. Find the Po10 for dx
    base, exp = power_of(dx, 1)
    
    # 2. Set the spacing as multiples of 10^x
    spacing = (10 ** exp) * PRIMARY_SPACING
    
    # Normalize the fade factor
    # The `base` is a value spanning 1-9, so this will make it a value 0.0 - 1.0
    fade = 1 - ((base-1) / 9)

    # Secondary (smaller) grid
    secondary_spacing = spacing * SECONDARY_SPACING
    x_lines(ctx, height, x1, x2, secondary_spacing, scale, fade, show_label=False)
    y_lines(ctx, width, y1, y2, secondary_spacing, scale, fade, show_label=False)
    
    # Primary, labeled, grid
    ctx.set_source_rgb(*LINES_RGB)
    x_lines(ctx, height, x1, x2, spacing, scale)
    y_lines(ctx, width, y1, y2, spacing, scale)

    pattern = cairo.SurfacePattern(surface)
    
    # Keep the grid lines crisp
    pattern.set_filter(cairo.Filter.NEAREST)
        
    return pattern
