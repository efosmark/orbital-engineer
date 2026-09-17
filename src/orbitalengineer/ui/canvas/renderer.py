import cairo
from orbitalengineer.ui.model import main
from orbitalengineer.ui.audio.synth import ToneSynthController
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.gtk4 import GObject, Graphene

from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.client import ClientSocketConnection

class Renderer(GObject.GObject):
    app:main.AppModel = GObject.Property(type=object) # type:ignore
    camera = GObject.Property(type=object)
    orbital:ClientSocketConnection = GObject.Property(type=object) # type:ignore
    
    def __init__(self, view:main.AppModel, camera:pz.Camera2D, orbital:ClientSocketConnection, clock:SimClock, synth:ToneSynthController):
        super().__init__()
        self.app = view
        self.camera = camera
        self.orbital = orbital
        self.clock = clock
        self.synth = synth
        self.initialize()
        
    def get_cairo(self, snapshot, width, height):
        return snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))
    
    def do_draw(self, snapshot, width, height):
        cr = self.get_cairo(snapshot, width, height)
        self.draw(cr, width, height)

    # def do_snapshot(self, snapshot: Gtk.Snapshot):
    #     width = self.get_allocated_width()
    #     height = self.get_allocated_height()

    #     if self._cached_surface is None or \
    #        self._cached_surface.get_width() != width or \
    #        self._cached_surface.get_height() != height:
    #         self._cached_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

    def initialize(self):
        ...

    def draw(self, cr:cairo.Context, width:int, height:int):
        raise NotImplementedError()
