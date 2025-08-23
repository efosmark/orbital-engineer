import cairo
from orbitalengineer.ui.gtk4 import Gtk

MIN_ZOOM = 0.0005
MAX_ZOOM = 600.0

class Camera2D:
    def __init__(self):
        self.offset = [0.0, 0.0]  # World space position (camera center)
        self.zoom = 1.0

    def get_matrix(self, width, height):
        matrix = cairo.Matrix()
        matrix.translate(width / 2, height / 2)
        matrix.scale(self.zoom, self.zoom)
        matrix.translate(-self.offset[0], -self.offset[1])
        return matrix

    def get_inverse_matrix(self, width, height):
        matrix = self.get_matrix(width, height)
        matrix.invert()
        return matrix

    def screen_to_world(self, x, y, width, height):
        inv = self.get_inverse_matrix(width, height)
        return inv.transform_point(x, y)

    def world_to_screen(self, x, y, width, height):
        mat = self.get_matrix(width, height)
        return mat.transform_point(x, y)

    def zoom_at(self, screen_x, screen_y, width, height, factor):
        wx, wy = self.screen_to_world(screen_x, screen_y, width, height)
        #self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        self.zoom = self.zoom * factor
        
        wx2, wy2 = self.screen_to_world(screen_x, screen_y, width, height)
        self.offset[0] += wx - wx2
        self.offset[1] += wy - wy2

class Camera2DController:
    def __init__(self, widget, camera):
        self.widget = widget
        self.camera = camera
        self.pointer_x = 0
        self.pointer_y = 0
        self.drag_start_offset = (0, 0)
        self._last_pinch_scale = 1.0

        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self.on_motion)
        widget.add_controller(motion)

        drag = Gtk.GestureDrag.new()
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        widget.add_controller(drag)

        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self.on_scroll)
        widget.add_controller(scroll)

        zoom = Gtk.GestureZoom.new()
        zoom.connect("scale-changed", self.on_pinch_zoom)
        widget.add_controller(zoom)
        self.zoom_gesture = zoom

    def on_motion(self, controller, x, y):
        self.pointer_x = x
        self.pointer_y = y
        self.widget.queue_draw()

    def on_drag_begin(self, gesture, start_x, start_y):
        self._last_pinch_scale = 1.0
        self.drag_start_offset = (self.camera.offset[0], self.camera.offset[1])

    def on_drag_update(self, gesture, dx, dy):
        self.camera.offset[0] = self.drag_start_offset[0] - dx / self.camera.zoom
        self.camera.offset[1] = self.drag_start_offset[1] - dy / self.camera.zoom
        self.widget.queue_draw()

    def on_scroll(self, controller, dx, dy):
        if dy == 0:
            return
        direction = -1 if dy > 0 else 1
        factor = 1.1 ** direction
        w = self.widget.get_allocated_width()
        h = self.widget.get_allocated_height()
        self.camera.zoom_at(self.pointer_x, self.pointer_y, w, h, factor)
        self.widget.queue_draw()
    
    
    def on_pinch_zoom(self, gesture, scale):
        if scale == 0:
            return  # Sanity check

        scale_delta = scale / self._last_pinch_scale
        self._last_pinch_scale = scale

        if abs(scale_delta - 1.0) < 0.001:
            return  # Ignore trivial changes

        sequences = gesture.get_sequences()
        _, x1, y1 = gesture.get_point(sequences[0])
        _, x2, y2 = gesture.get_point(sequences[1])
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        w = self.widget.get_allocated_width()
        h = self.widget.get_allocated_height()
        self.camera.zoom_at(x, y, w, h, scale_delta)
        self.widget.queue_draw()
