from orbitalengineer.ui.gtk4 import Gtk, GLib

import cairo
import math
import time

COLOR_BACKGROUND = (0.05, 0.05, 0.05)

COLOR_MEAN_ANOMALY = (0.8, 0.8, 0.8)
COLOR_ECCENTRIC_ANOMALY = (0.5, 0.5, 1.0)
COLOR_TRUE_ANOMALY = (1.0, 0.2, 0.2)
COLOR_ORBIT = (0.2, 1.0, 0.6)
COLOR_CENTRAL_BODY = (1,0.7,0.1)


def solve_kepler(M, e, tol=1e-6, max_iter=10):
    E = M
    for _ in range(max_iter):
        delta = E - e * math.sin(E) - M
        if abs(delta) < tol:
            break
        E -= delta / (1 - e * math.cos(E))
    return E


def draw_anomaly_meter(cr:cairo.Context, cx, cy, radius, M, *, color=(0.1, 0.1, 0.1), label:str|None=None):
    """
    Draws a "clock-style" hand for mean anomaly (M) at the center (cx, cy).

    Parameters:
        cr     - Cairo context
        cx, cy - center of the clock
        radius - length of the hand
        M      - mean anomaly (radians)
    """
    # Draw circle (optional, for visual clarity)
    cr.save()
    cr.translate(cx,cy)

    cr.new_path()
    cr.set_source_rgb(0.1, 0.1, 0.1)
    cr.arc(0,0, radius, 0, 2 * math.pi)
    cr.fill()
    
    r, g, b = color[:3]
    
    cr.set_line_width(4)
    cr.set_source_rgba(r,g,b, 0.8)
    cr.arc(0,0, radius, 0, 2 * math.pi)
    cr.stroke()

    # Draw hand
    M = -M
    x_end = radius * math.cos(M)
    y_end = radius * math.sin(M)
    cr.move_to(0,0)
    cr.line_to(x_end, y_end)
    cr.stroke()
    
    #cr.set_source_rgb(1, 1, 1)
    if label:
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)
        
        te = cr.text_extents(label)
        cr.move_to(0 - (te.width / 2), radius + 20)
        cr.show_text(label)
    
    cr.restore()

def draw_crosshair(cr, x, y, min_size, max_size):
    cr.set_line_width(1)
    
    cr.save()
    cr.translate(x, y)
    
    #1
    cr.move_to(0, -max_size)
    cr.line_to(0, -min_size)
    
    #1
    cr.move_to(0, max_size)
    cr.line_to(0, min_size)
    
    #3
    cr.move_to(-max_size, 0)
    cr.line_to(-min_size, 0)
    
    #4
    cr.move_to(max_size, 0)
    cr.line_to(min_size, 0)
    
    cr.set_source_rgba(0.95, 0.95, 0.95, 1)
    cr.stroke()
    cr.restore()

def draw_ellipse(cr, cx, cy, a, b, *, show_semimajor_axis=True):
    cr.new_path()
    cr.save()
    cr.translate(cx, cy)

    cr.scale(a, b)
    cr.arc(0, 0, 1, 0, 2 * math.pi)
    cr.restore()

    cr.stroke()
    
    if show_semimajor_axis:
        # Draw the major axis
        cr.move_to(cx-a,cy)
        cr.line_to(cx+a, cy)
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.3)
        cr.set_dash([8.0, 4.0])
        cr.stroke()
        cr.set_dash([])

def draw_auxiliary_circle(cr, cx, cy, a):
    cr.set_source_rgba(*COLOR_ECCENTRIC_ANOMALY, 0.3)
    cr.set_line_width(2)
    cr.arc(cx, cy, a, 0, 2 * math.pi)
    cr.set_dash([2.0, 4.0])
    cr.stroke()
    cr.set_dash([])

def draw_eccentric_anomaly(cr, cx, cy, x_circ, y_circ, E):
    # Eccentric anomaly arc (around ellipse center)
    cr.set_source_rgba(*COLOR_ECCENTRIC_ANOMALY, 0.3)
    cr.set_line_width(20)
    cr.arc_negative(cx, cy, 10, 0, E)
    cr.stroke()
    
    # Connecting line
    cr.set_source_rgba(*COLOR_ECCENTRIC_ANOMALY, 0.3)
    cr.set_line_width(2)
    cr.move_to(cx, cy)
    cr.line_to(x_circ, y_circ)
    cr.stroke()

def draw_true_anomaly(cr, fx, fy, x_true, y_true, nu):
    cr.set_line_width(2)
    cr.set_source_rgba(*COLOR_TRUE_ANOMALY, 0.3)
    cr.move_to(fx, fy)
    cr.line_to(x_true,  y_true)
    cr.stroke()
        

    # True anomaly arc (around focus)
    cr.set_source_rgba(*COLOR_TRUE_ANOMALY, 0.3)
    cr.set_line_width(10)
    cr.arc_negative(fx, fy, 20, 0, nu)
    cr.stroke()

def draw_projection_line(cr, x_proj, y_proj, x_circ, y_circ):
    cr.set_source_rgba(*COLOR_ORBIT, 0.2)
    cr.set_line_width(1)
    cr.set_dash([4.0, 4.0])
    cr.move_to(x_circ, y_circ)
    cr.line_to(x_proj, y_proj)
    cr.stroke()
    cr.set_dash([])

def draw_body(cr, x, y, radius, color):
    if len(color) == 3:
        color = [*color, 1]
    
    cr.set_source_rgba(*COLOR_BACKGROUND, 0.2)
    cr.arc(x, y, radius * 2, 0, 2 * math.pi)
    cr.fill()
    cr.set_source_rgba(*color)
    cr.arc(x, y, radius, 0, 2 * math.pi)
    cr.fill()

class EccentricAnomalyDiagram(Gtk.DrawingArea):
    def __init__(self):
        super().__init__()
        self.set_draw_func(self.on_draw)
        self.t0 = time.time()
        GLib.timeout_add(1000 // 60, self._queue_draw)

    def _queue_draw(self):
        self.queue_draw()
        return True

    def on_draw(self, area, cr, width, height):
        cr.set_source_rgb(*COLOR_BACKGROUND)
        cr.paint()
        
        t = -(time.time() - self.t0)
        M = (t * 0.5) % (2 * math.pi)

        a = 300  # semi-major axis
        e = 0.8  # eccentricity

        # Ellipse center
        cx = width // 2 #  + 60
        cy = height // 2
        
        # Focus center
        fx = cx  + (a * e)
        fy = cy

        # Eccentric anomaly
        E = solve_kepler(M, e)

        # semi-minor axis
        b = a * math.sqrt(1 - e ** 2)


        cr.set_source_rgba(*COLOR_ORBIT, 0.3)
        cr.set_line_width(2)
        draw_ellipse(cr, cx, cy, a, b)
        draw_auxiliary_circle(cr, cx, cy, a)
                
        # Eccentric anomaly line
        x_circ = cx + a * math.cos(E)
        y_circ = cy + a * math.sin(E)
        draw_eccentric_anomaly(cr, cx, cy, x_circ, y_circ, E)
        
        # Projection line
        x_proj = x_circ
        y_proj = cy + b * math.sin(E)
        draw_projection_line(cr, x_circ, y_circ, x_proj, y_proj)

        # True Anomaly
        nu = 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2),
            math.sqrt(1 - e) * math.cos(E / 2)
        )

        r = a * (1 - e**2) / (1 + e * math.cos(nu))
        x_true = fx + (r * math.cos(nu))
        y_true = fy + (r * math.sin(nu))

        draw_true_anomaly(cr, fx, fy, x_true, y_true, nu)

        # Focus (central mass)
        draw_body(cr, fx, fy, 10, color=COLOR_CENTRAL_BODY)

        # Orbiting body
        draw_body(cr, x_true, y_true, 5, color=COLOR_ORBIT)
        
        # Ghost orbit
        draw_body(cr, x_circ, y_circ, 5, color=(*COLOR_ECCENTRIC_ANOMALY, 0.4))
        
        # Apoapsis
        draw_body(cr, cx-a, cy, 5, color=(0.5,0.5,0.5, 0.5))
        draw_crosshair(cr, cx-a, cy, 10, 25)
        
        # Periapsis
        draw_body(cr, cx+a, cy, 5, color=(0.5,0.5,0.5, 0.5))
        draw_crosshair(cr, cx+a, cy, 10, 25)
        
        # Center point
        draw_crosshair(cr, cx, cy, 2, 25)
        
        # Focus
        draw_crosshair(cr, fx, fy, 15, 25)
        
        draw_anomaly_meter(
            cr, 80, 50, 30, -M,
            label=f"Mean Anomaly (M={(2*math.pi) - M:.1f})",
            color=COLOR_MEAN_ANOMALY
        )
        
        draw_anomaly_meter(
            cr, 80, 150, 30, -E,
            label=f"Ecc. Anomaly (E={(2*math.pi) - E:.1f})",
            color=COLOR_ECCENTRIC_ANOMALY
        )
        
        draw_anomaly_meter(
            cr, 80, 250, 30, -nu,
            label=f"True Anomaly (ν={(2*math.pi) - nu:.1f})",
            color=COLOR_TRUE_ANOMALY
        )
        

class DiagramApp(Gtk.Application):
    
    def __init__(self):
        super().__init__()

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("Orbital Anomalies with Angle Sweeps")
        win.set_default_size(800, 600)
        win.set_child(EccentricAnomalyDiagram())
        win.present()
        win.maximize()

if __name__ == "__main__":
    app = DiagramApp()
    app.run([])
