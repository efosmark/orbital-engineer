from typing import cast
import cairo

from orbitalengineer.ui.fmt import mag_format

X_PADDING = 10
X_SPACING = 30

Y_PADDING = 4

FONT_SIZE = 10
BG_COLOR = (0, 0, 0.1)
TEXT_COLOR = (0.55, 0, 0.45)


def create_debug_info(accum, body_count, fps, frame_no, tick_per_sec, steps_per_tick, width, height):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_hairline(True)
    ctx.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(FONT_SIZE)

    if accum > 1000:
        accum = mag_format(accum, sig=0)
    else:
        accum = f"{accum:.0f}"
    
    labels = [
        f"accum={accum}",
        f"{body_count:.0f} bodies",
        f"{fps=:.1f}",
        f"{tick_per_sec=:.0f}",
        f"{steps_per_tick=:.0f}",
        f"[T+{frame_no:.0f}]"
    ]
    
    r_height = max([
        ctx.text_extents(l).height for l in labels
    ]) + (Y_PADDING * 2)
    
    r_width = sum([
        ctx.text_extents(l).x_advance for l in labels
    ]) + (X_PADDING * 2) + (X_SPACING * (len(labels) - 1))
        
    ctx.save()
    ctx.translate(
        width - r_width,
        height - r_height
    )

    # Background
    if BG_COLOR is not None:
        ctx.set_source_rgb(*cast(tuple,BG_COLOR))
        ctx.rectangle(0, 0, r_width, r_height)
        ctx.fill()
    
    ctx.set_source_rgb(*TEXT_COLOR)
    baseline = r_height - Y_PADDING
    x = r_width - X_PADDING
    for label in reversed(labels):
        te = ctx.text_extents(label)
        x -= te.x_advance
        ctx.move_to(x, baseline)
        ctx.show_text(label)
        x -= X_SPACING

    ctx.restore()
    return surface
