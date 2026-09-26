from collections import deque
import time
from gi.repository import Gst #  type:ignore

from orbitalengineer.engine.orbitalcl.device import DeviceStatus
from orbitalengineer.ipc.message import HostStatus
from orbitalengineer.ui.audio import tone
from orbitalengineer.ui.canvas import renderer, sparkline

SHOW_DEVICE_TEMPERATURE = False
CRIT_TONE = tone.OnOff(((tone.Sine(440)) * 3), tone.PWM(2, 0.40))

MARGIN = 35

GRAPH_HEIGHT = 40
GRAPH_WIDTH = 140

MX_NUM_DATA_POINTS = 60

CRITICAL_UTILIZATION = 70.0

COLOR_HEX_HOST_UTIL_CRIT_BG = "#3137A355"
COLOR_HEX_HOST_UTIL_BG = "#56359D2C"
COLOR_HEX_HOST_UTIL_LINE = "#3509C7AC"
COLOR_HEX_HOST_UTIL_LABEL = "#9C8ECBAC"

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


class ResourceMonitorRenderer(renderer.Renderer):
    
    def initialize(self):
        self.device_history:deque[DeviceStatus] = deque(maxlen=MX_NUM_DATA_POINTS)
        self.cpu_history:deque[HostStatus] = deque(maxlen=MX_NUM_DATA_POINTS)
        self.last_record_time = time.monotonic()
    
    def is_temperature_critical(self):
        if len(self.device_history) == 0:
            return False
        latest = self.device_history[-1]
        return latest.temperature is not None and latest.temperature >= CRITICAL_TEMPERATURE_C

    def is_utilization_critical(self):
        if len(self.device_history) == 0:
            return False
        latest = self.device_history[-1]
        return latest.utilization is not None and latest.utilization >= CRITICAL_UTILIZATION

    
    def draw(self, cr, width:int, height:int):
        if not self.app.show_debug_info:
            return
        
        if self.orbital.gpu_status is None:
            return
        
        t = time.monotonic()
        if t >= self.last_record_time + 0.25:
            self.device_history.append(self.orbital.gpu_status)
            self.cpu_history.append(self.orbital.cpu_status)
            self.last_record_time = t

        cr.save()

        graph_start_x = width - GRAPH_WIDTH - MARGIN
        graph_start_y = height - GRAPH_HEIGHT - MARGIN

        critical = self.is_utilization_critical()
        flash_critical = critical and (int(time.monotonic()) % 2 == 0) 
        sparkline.draw_graph(
            cr,
            MX_NUM_DATA_POINTS,
            [g.utilization for g in self.device_history],
            graph_start_x,
            graph_start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format="Device Util {value:.0f}%",
            label_color_hex=COLOR_HEX_UTIL_LABEL,
            bg_color_hex=COLOR_HEX_UTIL_BG if not flash_critical else COLOR_HEX_UTIL_CRIT_BG,
            line_color_hex=COLOR_HEX_UTIL_LINE,
            y_min=0.0,
            y_max=100.0
        )
        
        graph_start_y = graph_start_y - GRAPH_HEIGHT - 5
        sparkline.draw_graph(
            cr,
            MX_NUM_DATA_POINTS,
            [c.utilization for c in self.cpu_history],
            graph_start_x,
            graph_start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format=f"Host Util {{value:.0f}}%   (core {self.orbital.cpu_status.id})",
            label_color_hex=COLOR_HEX_HOST_UTIL_LABEL,
            bg_color_hex=COLOR_HEX_HOST_UTIL_BG,
            line_color_hex=COLOR_HEX_HOST_UTIL_LINE,
            y_min=0,
            y_max=100.0
        )
        
        if SHOW_DEVICE_TEMPERATURE:    
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
                [g.temperature for g in self.device_history],
                graph_start_x,
                graph_start_y,
                width=GRAPH_WIDTH,
                height=GRAPH_HEIGHT,
                label_format="Device Temp {value:.0f} °C",
                label_color_hex=COLOR_HEX_TEMP_LABEL,
                bg_color_hex=COLOR_HEX_TEMP_BG if not flash_critical else COLOR_HEX_TEMP_CRIT_BG,
                line_color_hex=COLOR_HEX_TEMP_LINE,
                y_min=30.0,
                y_max=100.0
            )

        cr.restore()