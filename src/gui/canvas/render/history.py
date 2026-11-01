import cmath
from collections import defaultdict, deque
import cairo
from gui.canvas import renderer
from gui.ui_config import NUM_HISTORY_POSITIONS, HIST_MIN_ANGLE_TO_RECORD

import numpy as np

HISTORY_PATH_ALPHA = 0.15
HISTORY_PATH_LINE_WIDTH = 4.0

def circumcircle(p1: complex, p2: complex, p3: complex):
    """ Compute he circumcenter and circumradius of three points.
    Finds the center and radius of the unique circle that passes through all three points.
    """
    v1 = p1 - p2
    v2 = p1 - p3

    A, B = v1.real, v1.imag
    C, D = v2.real, v2.imag

    E = (abs(p1)**2 - abs(p2)**2) / 2
    F = (abs(p1)**2 - abs(p3)**2) / 2

    det = A * D - B * C
    if abs(det) < 1: # TODO change to a scalable value
        return None

    center = complex(
        (D * E - B * F) / det,
        (A * F - C * E) / det
    )

    # Euclidiean distance
    r = abs(center - p1)
        
    a1 = np.atan2(p1.imag - center.imag, p1.real - center.real)
    a2 = np.atan2(p2.imag - center.imag, p2.real - center.real)
    a3 = np.atan2(p3.imag - center.imag, p3.real - center.real)
    
    cross = ((p2 - p1).conjugate() * (p3 - p1)).imag
    is_negative = cross < 0
    
    return (center, r, a1, a2, a3, is_negative)

def should_record(last_velocity:complex, curr_velocity:complex) -> bool:
    _, last_angle = cmath.polar(last_velocity)
    _, curr_angle = cmath.polar(curr_velocity)
    return abs(np.degrees(curr_angle - last_angle)) >= HIST_MIN_ANGLE_TO_RECORD

class HistoryRenderer(renderer.Renderer):
    hp = defaultdict(lambda:deque(maxlen=NUM_HISTORY_POSITIONS))
    hv = defaultdict(lambda:deque(maxlen=NUM_HISTORY_POSITIONS))
    midpoints = defaultdict(lambda:deque(maxlen=NUM_HISTORY_POSITIONS))
    body_id:int|None = None
    
    def initialize(self):
        self.app.engine.connect('notify::tick-id', self.__on_tick_changed)
        self.app.connect('notify::secondary-body', self.__on_tracked_body_changed)

    def __on_tracked_body_changed(self, m, param):
        if self.body_id != self.app.secondary_body:
            self.body_id = self.app.secondary_body
            self.hp.clear()
            self.hv.clear()
            self.midpoints.clear()

    def __on_tick_changed(self, m, param):
        body_id = self.body_id
        if body_id is None:return
        
        particle = self.app.engine.get_particle(self.app.secondary_body)
        if particle.is_removed:
            self.body_id = None
            return

        if len(self.hp[body_id]) < 3:
            self.hp[body_id].append(particle.position)
            self.hv[body_id].append(particle.velocity)
            return

        if not should_record(self.hv[body_id][-1], particle.velocity):
            return

        self.hp[body_id].append(particle.position)
        self.hv[body_id].append(particle.velocity)

        result = circumcircle(self.hp[body_id][-3], self.hp[body_id][-2], self.hp[body_id][-1])
        if result:
            self.midpoints[self.app.secondary_body].append(result)

    def draw_history(self, cr:cairo.Context, body_id:int):
        particle = self.app.engine.get_particle(body_id)
        if particle.is_removed: return
        
        cr.set_line_width(HISTORY_PATH_LINE_WIDTH/self.camera.zoom)
        
        r,g,b,_ = self.app.particle_colors.get(body_id, (1, 1, 1, 1))
        cr.set_source_rgba(r, g, b, HISTORY_PATH_ALPHA)
            
        for (center, r, a1, a2, a3, is_negative) in self.midpoints[body_id]:
            cr.new_path()
            if not is_negative:
                cr.arc(center.real, center.imag, r, a2, a3)
            else: 
                cr.arc_negative(center.real, center.imag, r, a2, a3)
            cr.stroke()
        
        # Connect to the current position
        if len(self.hp[body_id]) > 0:
            last_position = self.hp[body_id][-1]
            cr.new_path()
            cr.move_to(last_position.real, last_position.imag)
            cr.line_to(particle.position.real, particle.position.imag)
            cr.stroke()

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.app.show_focused_history: return
        if self.app.show_focused_history and self.app.secondary_body is not None:
            self.draw_history(cr, int(self.app.secondary_body))