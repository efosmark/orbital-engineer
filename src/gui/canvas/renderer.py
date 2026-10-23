import cairo
from gui.model import main
from gui.canvas import pz
from gui.gtk4 import GObject, Graphene

from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.client import ClientSocketConnection

class Renderer(GObject.GObject):
    app:main.AppModel = GObject.Property(type=object) # type:ignore
    camera = GObject.Property(type=object)
    orbital:ClientSocketConnection = GObject.Property(type=object) # type:ignore
    
    def __init__(self, view:main.AppModel, camera:pz.Camera2D, orbital:ClientSocketConnection, clock:SimClock):
        super().__init__()
        self.app = view
        self.camera = camera
        self.orbital = orbital
        self.clock = clock
        self.initialize()
        
    def get_cairo(self, snapshot, width, height):
        return snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

    def initialize(self):
        ...

    def draw(self, cr:cairo.Context, width:int, height:int):
        raise NotImplementedError()
