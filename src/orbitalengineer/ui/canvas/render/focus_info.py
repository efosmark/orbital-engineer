from cmath import polar
import numpy as np
import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.gtk4 import Gtk, Graphene
from orbitalengineer.ui.fmt import mag_format, positive_angle

X_PADDING = 10
X_SPACING = 30

Y_PADDING = 4

FONT_SIZE = 8
BG_COLOR = (0.05, 0, 0.05, 0.4)
TEXT_COLOR = (0.65, 0.1, 0.55)

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
        y += LINE_HEIGHT
        ctx.move_to(x, y)
        ctx.show_text(line)
        y += LINE_SPACING
    
    return rec


def create_panel(cr, title:str|None, lines:list[str], fixed_width:None|int=None):
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
    height = int(full_title_height + full_content_height + Y_PAD)
        
    # Background
    cr.set_source_rgb(*BG_COLOR)
    cr.rectangle(0, 0, width, height)
    cr.fill()
    
    # Border
    cr.set_source_rgb(*FG_COLOR)
    cr.set_hairline(True)
    cr.rectangle(0, 0, width, height)
    cr.stroke()
    
    # Title
    if rec_title:
        cr.set_source_rgb(*FG_COLOR)
        cr.rectangle(0, 0, width, full_title_height)
        cr.fill()
        cr.set_source_surface(rec_title, X_PAD, title_h + Y_PAD)
        cr.paint()
    
    cr.set_source_surface(rec_content, X_PAD, full_title_height)
    cr.paint()

 
class FocusInfoRenderer(renderer.Renderer):
    
    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.view.show_focus_info:
            return
        
        if self.view.hovered_over_particle is not None:
            b = self.view.hovered_over_particle
        elif self.view.secondary_body is not None:
            b = self.orbital.get_particle(self.view.secondary_body)
        else:
            return
        
        disp = [
            ("mass",     f"{mag_format(b.get_mass())} kg"),
            ("radius",   f"{mag_format(b.get_radius())} m"),
         ]
        
        velocity = b.get_velocity()
        mag, angle = polar(velocity)
        angle_degrees = np.degrees(positive_angle(angle))
        
        # min_dt = b.get_min_toi()
        # if np.isinf(min_dt):
        #     min_dt = '--.------'
        # else:
        #     min_dt = f"{min_dt:.6f}"
        
        disp.extend([
            ("vel mag",   f"{mag_format(mag)} m/s"),
            ("vel angle",      f"{angle_degrees:.1f}°"),
            #("dt",         min_dt)
        ])
        
        cr.translate(20, 20)
        
        name = self.view.particle_names.get(b.idx, f"{b.idx}")
        create_panel(cr, f"{name}-{b.idx}", [
            f"{label:<8} {value:>18}"
            for label, value in disp
        ])