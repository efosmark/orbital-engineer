import math
import numpy as np
import cairo
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.fmt import positive_angle

DEFAULT_RETICLE_COLOR = (1, 1, 1, 0.3)


def from_polar(theta, mag) -> complex:
    x = np.cos(theta) * mag
    y = np.sin(theta) * mag
    return complex(x, y)

def show_text_flipped(cr:cairo.Context, text:str):
    cr.save()
    te = cr.text_extents(text)

    # offset
    cr.translate(te.width, -te.height)
        
    # flip
    cr.rotate(np.pi)
    
    cr.show_text(text)
    cr.restore()
    
    
class ReticleRenderer(renderer.Renderer):

    def draw_tick_mark(self, cr:cairo.Context, angle_deg, radius, show_label, highlight:bool=False):
            reticle_start = radius + radius/10
            
            if show_label:
                reticle_end = reticle_start + radius/5
            else:
                reticle_end = reticle_start + radius/10
            
            cr.save()
            cr.move_to(reticle_start, 0)
            cr.line_to(reticle_end, 0)
            cr.stroke()
            
            if show_label:
                deg_label = f"{angle_deg:.0f}°"
                
                cr.save()
                off = -cr.text_extents(deg_label).y_bearing/2
                cr.translate(reticle_end + (radius/10), off)            
                if angle_deg > 90 and angle_deg < 265:
                    show_text_flipped(cr, deg_label)
                else:
                    cr.show_text(deg_label)
                cr.restore()
            
            cr.restore() 

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.data.secondary_body:
            return
        
        body = self.orbital.get_particle(self.data.secondary_body)

        cr.select_font_face(self.view.font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(self.view.font_size/self.camera.zoom)

        cr.set_line_width(1/self.camera.zoom)
        cr.set_source_rgba(*DEFAULT_RETICLE_COLOR)
        radius = body.get_radius()
        
        pos = body.get_position()
        x, y = pos.real, pos.imag
        
        try:
            cr.translate(x, y)
        except cairo.Error:
            print(f"ERROR. Cannot translate {x} and {y}")
            cr.restore()
            return
                
        if radius * self.camera.zoom < 5:
            return
        
        elif radius * self.camera.zoom < 20:
            return
        
        cr.set_source_rgba(*DEFAULT_RETICLE_COLOR)
        
        reticle_count = 64
        total_angle = 0
        for i in range(reticle_count):
            angle = (2.0*np.pi) / reticle_count
            angle_deg = round(np.degrees(total_angle), 2)
            self.draw_tick_mark(cr, angle_deg, radius, angle_deg % 45 == 0)
                
            cr.rotate(angle)
            total_angle += angle

        velocity = body.get_velocity()
        angle = positive_angle(np.atan2(velocity.imag, velocity.real))

        cr.save()
        cr.rotate(angle)
        cr.set_line_width(3/self.camera.zoom)
        cr.set_source_rgba(1, 1, 1)
        self.draw_tick_mark(cr, np.degrees(angle), radius, True, True)
        cr.restore()