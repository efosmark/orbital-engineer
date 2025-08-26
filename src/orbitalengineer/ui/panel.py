import cairo

BG_COLOR = (0.1, 0.1, 0.1)
FG_COLOR = (0.9, 0.9, 0.9)
FONT_FAMILY = "Liberation Mono"

TITLE_FONT_SIZE = 11
CONTENT_FONT_SIZE = 11

X_PAD = 4
Y_PAD = 4
LINE_SPACING = 0
LINE_HEIGHT = CONTENT_FONT_SIZE

def create_background(width:int, height:int) -> cairo.RecordingSurface:
    rec = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(rec)
    
    # Background
    ctx.set_source_rgb(*BG_COLOR)
    ctx.paint()
    
    # Draw the border
    ctx.set_source_rgb(*FG_COLOR)
    ctx.set_hairline(True)
    ctx.rectangle(0, 0, width, height)
    ctx.stroke()
    return rec


def create_title(title:str):
    rec = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(rec)
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(TITLE_FONT_SIZE)
    ctx.set_source_rgb(*BG_COLOR)
    ctx.show_text(title.upper())
    return rec


def create_contents(lines:list[str]):
    rec = cairo.RecordingSurface(cairo.Content.COLOR_ALPHA, None)
    ctx = cairo.Context(rec)

    x, y = 0, 0

    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(CONTENT_FONT_SIZE)
    ctx.set_source_rgb(*FG_COLOR)

    y = y + Y_PAD
    for line in lines:
        y += LINE_HEIGHT #ctx.text_extents(line).height
        ctx.move_to(x, y)
        ctx.show_text(line)
        y += LINE_SPACING
    
    return rec


def create_panel(title:str|None, lines:list[str], fixed_width:None|int=None) -> cairo.ImageSurface:
    rec_title = None
    if title:
        rec_title = create_title(title)
        _, _, title_w, title_h = rec_title.ink_extents()
        full_title_height = title_h + (Y_PAD * 2)
    
    else:
        title_w, title_h, full_title_height = 0, 0, 0
    
    rec_content = create_contents(lines)
    content_x, content_y, content_w, content_h = rec_content.ink_extents()
    full_content_height = content_h + (Y_PAD * 2)
    
    width = int(max(title_w, content_w, fixed_width or 0)) + (X_PAD * 2)
    height = int(full_title_height + full_content_height)
    
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(img)
    
    # Background
    ctx.set_source_rgb(*BG_COLOR)
    ctx.paint()
    
    # Border
    ctx.set_source_rgb(*FG_COLOR)
    ctx.set_hairline(True)
    ctx.rectangle(0, 0, width, height)
    ctx.stroke()
    
    # Title
    if rec_title:
        ctx.set_source_rgb(*FG_COLOR)
        ctx.rectangle(0, 0, width, full_title_height)
        ctx.fill()
        ctx.set_source_surface(rec_title, X_PAD, title_h + Y_PAD)
        ctx.paint()
    
    ctx.set_source_surface(
        rec_content,
        -content_x + X_PAD,
        -content_y + Y_PAD + full_title_height
    )
    ctx.paint()
    
    return img