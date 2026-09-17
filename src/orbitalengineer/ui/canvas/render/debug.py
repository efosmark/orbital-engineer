import math
import statistics
from dataclasses import dataclass
from typing import cast
import cairo

from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.gtk4 import Gdk

X_PADDING = 5
X_SPACING = 30
X_MARGIN = 35

Y_PADDING = 10
Y_SPACING = 0
Y_MARGIN = 5

FONT_SIZE = 9
BG_COLOR = (0.05, 0, 0.05, 0.7)
TEXT_COLOR = (0.7, 0.7, 0.7)
CRITICAL_TEXT_COLOR = (0.8, 0.2, 0.2)
WARNING_TEXT_COLOR = (0.8, 0.7, 0.2)
BORDER_COLOR = (0.1, 0.1, 0.1)

NOMINAL = 0
WARNING = 1
CRITICAL = 2

ThresholdProfile_T = list[tuple[int, int]]

RENDER_MS_THRESHOLD:ThresholdProfile_T = [
    (0,    NOMINAL),
    (100,  WARNING),
    (150,  CRITICAL)
]

TICK_RATE_THRESHOLD:ThresholdProfile_T = [
    (0,    CRITICAL),
    (15,   WARNING),
    (30,   NOMINAL)
]

FRAME_RATE_THRESHOLD:ThresholdProfile_T = [
    (0,    CRITICAL),
    (10,   WARNING),
    (15,   NOMINAL)
]

def get_threshold(value, tmap:list[tuple[float,int]]):
    for A,B in zip(tmap, [*tmap[1:], None]):
        if A[0] <= value and (B is None or value < B[0]):
            return A[1]
    return NOMINAL

def color_from_threshold(threshold):
    if threshold == WARNING:
        return WARNING_TEXT_COLOR
    elif threshold == CRITICAL:
        return CRITICAL_TEXT_COLOR
    return TEXT_COLOR

def get_color(value, tmap):
    return color_from_threshold(get_threshold(value, tmap))

Color_T = tuple[float,float,float]

@dataclass
class DebugDisplayField:
    label:str
    value:int|float|str
    precision:int = 0
    unit:str|None = None
    threhold_profile:ThresholdProfile_T|None = None

class DebugInfoRenderer(renderer.Renderer):

    def _get_labels(self):
        try:
            num_bodies = self.app.engine.N
        except AttributeError:
            num_bodies = 0
                
        zoom = self.camera.zoom
        if zoom < 1: zoom = -1/zoom
        
        display:list[DebugDisplayField|None] = []
        display.extend([
            DebugDisplayField("Bodies", f"{self.orbital.get_valid_indices().size} / {int(num_bodies)}"),
            DebugDisplayField("Zoom",   zoom, 1),
        ])

        render_durations = self.app.durations['render'][-10:]
        if len(render_durations) > 0:
            t_render_ms = statistics.mean([d[1] for d in render_durations]) * 1000.0
            display.append(DebugDisplayField("Render", t_render_ms, 2, 'ms', RENDER_MS_THRESHOLD))
        
        frame_clock = cast(Gdk.FrameClock, self.app.frame_clock)
        fps = frame_clock.get_fps()
        frame_no = self.app.frame_clock.get_frame_counter()
        frame_interval_ms = (1/(fps or 1)) * 1000.0
        
        display.extend([
            None,
            DebugDisplayField('Frame #',  frame_no),
            DebugDisplayField("Rate",     fps, 1, "/s", FRAME_RATE_THRESHOLD),
            DebugDisplayField("Interval", frame_interval_ms, 1, 'ms')
        ])

        try:
            display.extend([
                None,
                DebugDisplayField('Tick #',   self.orbital.tick_id)
            ])
        except AttributeError:
            pass
        
        tick_durations = self.app.durations['tick'][-10:]
        if len(tick_durations) > 0:
            t_tick = statistics.mean([d[1] for d in tick_durations])
            t_tick_ms = t_tick * 1000.0
            tick_rate = 1/t_tick
            
            display.extend([
                DebugDisplayField("Rate",     tick_rate, 1, "/s", TICK_RATE_THRESHOLD),
                DebugDisplayField("Interval", t_tick_ms, 1, 'ms')
            ])

        return display
    
    
    def draw(self, cr, width:int, height:int):
        if not self.app.show_debug_info:
            return
        
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)
        
        labels = self._get_labels()

        ### ### ###

        r_height = (len(labels) * (FONT_SIZE + Y_SPACING)) + (Y_PADDING * 2)
        
        max_label_width = max([
          # cr.text_extents(l[0]).x_advance for l in labels
           cr.text_extents(l.label).x_advance for l in labels if l is not None

        ])
        
        max_value_width = 70
        # max_value_width = max([
        #    cr.text_extents(str(l.value)).x_advance for l in labels if l is not None
        # ])
        
        r_width = X_PADDING + max_label_width + X_SPACING + max_value_width
    
        cr.save()
        cr.translate(
            width - r_width - X_MARGIN,
            Y_MARGIN
        )

        # Background
        if BG_COLOR is not None:
            cr.set_source_rgb(*BORDER_COLOR)
            cr.rectangle(0, 0, r_width, r_height)
            cr.stroke()
            cr.set_source_rgba(*cast(tuple, BG_COLOR))
            cr.rectangle(0, 0, r_width, r_height)
            cr.fill()
        
        
        max_v_int_width = max([
            int(math.log10(abs(d.value) or 1))
            for d in labels
            if d is not None and isinstance(d.value, (int, float))
        ]) + 1
            
        y = Y_PADDING + FONT_SIZE
        for d in labels:
            if d is None:
                y += FONT_SIZE + Y_SPACING
                continue
            
            if d.threhold_profile is None:
                cr.set_source_rgb(*TEXT_COLOR)
            else:
                cr.set_source_rgb(*get_color(d.value, d.threhold_profile))
            
            # Right-aligned label
            cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            label_width = cr.text_extents(d.label).width
            cr.move_to(X_PADDING + (max_label_width - label_width), y)
            cr.show_text(d.label)
            
            # Decimal-aligned value
            if isinstance(d.value, str):
                value = d.value
            else:
                v_str = str(float(d.value))
                v_int, v_frac = v_str.split('.', 1)
                v_frac = v_frac[:d.precision]
                value = f"{v_int:>{max_v_int_width}}{'.' if d.precision > 0 else ' '}{v_frac} {d.unit or ''}"
            cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.move_to(max_label_width + X_SPACING, y)
            cr.show_text(value)
            
            # Carriage return
            y += FONT_SIZE + Y_SPACING    
        cr.restore()
