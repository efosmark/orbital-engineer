import cairo
from orbitalengineer.ui import model
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.gtk4 import GObject, Graphene

from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.client import ClientSocketConnection

class Renderer(GObject.GObject):
    
    view = GObject.Property(type=object)
    camera = GObject.Property(type=object)
    orbital:ClientSocketConnection = GObject.Property(type=object) # type:ignore
    
    def __init__(self, view:model.ViewModel, camera:pz.Camera2D, orbital:ClientSocketConnection, clock:SimClock):
        super().__init__()
        self.view = view
        self.camera = camera
        self.orbital = orbital
        self.clock = clock
        
    def get_cairo(self, snapshot, width, height):
        return snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))
    
    def do_draw(self, snapshot, width, height):
        cr = self.get_cairo(snapshot, width, height)
        self.draw(cr, width, height)

    def draw(self, cr:cairo.Context, width:int, height:int):
        raise NotImplementedError()
