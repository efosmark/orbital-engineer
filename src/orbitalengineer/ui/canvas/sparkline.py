import cairo
from orbitalengineer.ui import ui_config
from orbitalengineer.ui.color import set_source_hex

FONT_SIZE = 8
LABEL_PADDING = 2

def rescale(val:float, v_min, v_max, scale_max) -> float:
    v_range = v_max - v_min
    v_normalized = (val - v_min) / v_range
    return v_normalized * scale_max

def draw_graph(
    cr:cairo.Context,
    N:int,
    values:list[float|None],
    x:float,
    y:float,
    width:float,
    height:float,
    label_format:str,
    label_color_hex:str,
    bg_color_hex:str,
    line_color_hex:str,
    y_min:float,
    y_max:float
):
    cr.save()
    cr.translate(x, y)
    cr.move_to(0, 0)
    
    cr.set_hairline(True)
    set_source_hex(cr, bg_color_hex)
    cr.rectangle(0, 0, width, height)
    cr.fill()

    set_source_hex(cr, line_color_hex)

    valid_values = [v for v in values if v is not None]
    
    pitch = width / (N - 1)
    offset = N - len(valid_values)
    p0 = None
    for i,p1 in enumerate(valid_values):
        if p0 is None:
            p0 = p1
            continue
        
        i = offset + i
        
        a = rescale(p0, y_min, y_max, height)
        b = rescale(p1, y_min, y_max, height)
        
        cr.move_to((i - 1) * pitch, height - a)
        cr.line_to((i) * pitch, height - b)
        cr.stroke()
        
        p0 = p1
    
    if p0 is not None:
        cr.select_font_face(ui_config.DEFAULT_FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(FONT_SIZE)
        
        label = label_format.format(value=p0)
        te = cr.text_extents(label)
        cr.move_to(LABEL_PADDING, te.height + LABEL_PADDING)
        
        set_source_hex(cr, label_color_hex)
        cr.show_text(label)

    cr.restore()