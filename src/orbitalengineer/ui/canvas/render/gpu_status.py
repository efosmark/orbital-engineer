from collections import deque
import time
from gi.repository import Gst #  type:ignore

from orbitalengineer.engine.orbitalcl.device import GPUStatus
from orbitalengineer.ui.audio import tone
from orbitalengineer.ui.canvas import renderer, sparkline


CRIT_TONE = tone.OnOff(((tone.Sine(440)) * 3), tone.PWM(2, 0.40))

MARGIN = 35

GRAPH_HEIGHT = 40
GRAPH_WIDTH = 140

MX_NUM_DATA_POINTS = 60

CRITICAL_UTILIZATION = 70.0

COLOR_HEX_UTIL_CRIT_BG = "#5D981D57"
COLOR_HEX_UTIL_BG = "#436E182D"
COLOR_HEX_UTIL_LINE = "#A4C709AC"
COLOR_HEX_UTIL_LABEL = "#C0CB8EAC"

TEMPERATURE_MIN = 30.0
TEMPERATURE_MAX = 100.0
CRITICAL_TEMPERATURE_C = 90.0

COLOR_HEX_TEMP_CRIT_BG = "#981D1D5D"
COLOR_HEX_TEMP_BG = "#6F2A2A2F"
COLOR_HEX_TEMP_LINE = "#A93C2EAC"
COLOR_HEX_TEMP_LABEL = "#C69B95AC"


class GPUStatusRenderer(renderer.Renderer):
    
    def initialize(self):
        self.gpu_history:deque[GPUStatus] = deque(maxlen=MX_NUM_DATA_POINTS)
        self.last_record_time = time.monotonic()
    
    def is_temperature_critical(self):
        if len(self.gpu_history) == 0:
            return False
        latest = self.gpu_history[-1]
        return latest.temperature is not None and latest.temperature >= CRITICAL_TEMPERATURE_C

    def is_utilization_critical(self):
        if len(self.gpu_history) == 0:
            return False
        latest = self.gpu_history[-1]
        return latest.utilization is not None and latest.utilization >= CRITICAL_UTILIZATION

    
    def draw(self, cr, width:int, height:int):
        if not self.app.show_debug_info:
            return
        
        if self.orbital.gpu_status is None:
            return
        
        t = time.monotonic()
        if t >= self.last_record_time + 0.25:
            self.gpu_history.append(self.orbital.gpu_status)
            self.last_record_time = t

        cr.save()

        graph_start_x = width - GRAPH_WIDTH - MARGIN
        graph_start_y = height - GRAPH_HEIGHT - MARGIN

        critical = self.is_utilization_critical()
        flash_critical = critical and (int(time.monotonic()) % 2 == 0) 
        sparkline.draw_graph(
            cr,
            MX_NUM_DATA_POINTS,
            [g.utilization for g in self.gpu_history],
            graph_start_x,
            graph_start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format="GPU Util {value:.0f}%",
            label_color_hex=COLOR_HEX_UTIL_LABEL,
            bg_color_hex=COLOR_HEX_UTIL_BG if not flash_critical else COLOR_HEX_UTIL_CRIT_BG,
            line_color_hex=COLOR_HEX_UTIL_LINE,
            y_min=0.0,
            y_max=100.0
        )
            
        graph_start_y = graph_start_y - GRAPH_HEIGHT - 5
        
        critical = self.is_temperature_critical()
        if critical:
            if CRIT_TONE not in self.synth.tones:
                self.synth.tones.append(CRIT_TONE)
            self.synth.pipeline.set_state(Gst.State.PLAYING)
        else:
            if CRIT_TONE in self.synth.tones:
                self.synth.tones.remove(CRIT_TONE)
            self.synth.pipeline.set_state(Gst.State.PAUSED)
        
        flash_critical = critical and (int(time.monotonic()) % 2 == 0) 
        
        sparkline.draw_graph(
            cr,
            MX_NUM_DATA_POINTS,
            [g.temperature for g in self.gpu_history],
            graph_start_x,
            graph_start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format="GPU Temp {value:.0f} °C",
            label_color_hex=COLOR_HEX_TEMP_LABEL,
            bg_color_hex=COLOR_HEX_TEMP_BG if not flash_critical else COLOR_HEX_TEMP_CRIT_BG,
            line_color_hex=COLOR_HEX_TEMP_LINE,
            y_min=30.0,
            y_max=100.0
        )

        cr.restore()