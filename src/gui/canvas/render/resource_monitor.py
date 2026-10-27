from collections import deque
import time

from orbitalengineer.engine.device import DeviceStatus
from orbitalengineer.ipc.message import HostStatus
from gui.canvas import sparkline, renderer

MARGIN = 35

GRAPH_HEIGHT = 40
GRAPH_WIDTH = 140

MAX_NUM_DATA_POINTS = 60

CRITICAL_UTILIZATION = 70.0

COLOR_HEX_HOST_UTIL_CRIT_BG = "#3137A355"
COLOR_HEX_HOST_UTIL_BG = "#56359D2C"
COLOR_HEX_HOST_UTIL_LINE = "#3509C7AC"
COLOR_HEX_HOST_UTIL_LABEL = "#9C8ECBAC"

COLOR_HEX_UTIL_CRIT_BG = "#5D981D57"
COLOR_HEX_UTIL_BG = "#436E182D"
COLOR_HEX_UTIL_LINE = "#A4C709AC"
COLOR_HEX_UTIL_LABEL = "#C0CB8EAC"

COLOR_HEX_TEMP_CRIT_BG = "#981D1D5D"
COLOR_HEX_TEMP_BG = "#6F2A2A2F"
COLOR_HEX_TEMP_LINE = "#A93C2EAC"
COLOR_HEX_TEMP_LABEL = "#C69B95AC"


class ResourceMonitorRenderer(renderer.Renderer):
    
    def initialize(self):
        self.device_history:deque[DeviceStatus] = deque(maxlen=MAX_NUM_DATA_POINTS)
        self.cpu_history:deque[HostStatus] = deque(maxlen=MAX_NUM_DATA_POINTS)
        self.last_record_time = time.monotonic()

    def is_utilization_critical(self):
        if len(self.device_history) == 0:
            return False
        latest = self.device_history[-1]
        return latest.utilization is not None and latest.utilization >= CRITICAL_UTILIZATION
    
    def draw_device_utilization(self, cr, start_x, start_y):
        critical = self.is_utilization_critical()
        flash_critical = critical and (int(time.monotonic()) % 2 == 0) 
        sparkline.draw_graph(
            cr,
            MAX_NUM_DATA_POINTS,
            [g.utilization for g in self.device_history],
            start_x,
            start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format="Device Util {value:.0f}%",
            label_color_hex=COLOR_HEX_UTIL_LABEL,
            bg_color_hex=COLOR_HEX_UTIL_BG if not flash_critical else COLOR_HEX_UTIL_CRIT_BG,
            line_color_hex=COLOR_HEX_UTIL_LINE,
            y_min=0.0,
            y_max=100.0
        )
    
    def draw_host_utilization(self, cr, start_x, start_y):
        sparkline.draw_graph(
            cr,
            MAX_NUM_DATA_POINTS,
            [c.utilization for c in self.cpu_history],
            start_x,
            start_y,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            label_format=f"Host Util {{value:.0f}}%   (core {self.app.engine.cpu_status.id})",
            label_color_hex=COLOR_HEX_HOST_UTIL_LABEL,
            bg_color_hex=COLOR_HEX_HOST_UTIL_BG,
            line_color_hex=COLOR_HEX_HOST_UTIL_LINE,
            y_min=0,
            y_max=100.0
        )

    def draw(self, cr, width:int, height:int):
        if not self.app.show_debug_info:return
        
        t = time.monotonic()
        if t >= self.last_record_time + 0.25:
            if self.app.engine.gpu_status is not None:
                self.device_history.append(self.app.engine.gpu_status)
            if self.app.engine.cpu_status is not None:
                self.cpu_history.append(self.app.engine.cpu_status)
            self.last_record_time = t

        cr.save()
        graph_start_x = width - GRAPH_WIDTH - MARGIN
        graph_start_y = height - GRAPH_HEIGHT - MARGIN
        if self.app.engine.gpu_status is not None:
            self.draw_device_utilization(cr, graph_start_x, graph_start_y)
            graph_start_y = graph_start_y - GRAPH_HEIGHT - 5
        if self.app.engine.cpu_status is not None:
            self.draw_host_utilization(cr, graph_start_x, graph_start_y)
        cr.restore()