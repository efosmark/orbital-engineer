import cairo
from orbitalengineer.engine.simcontroller import OrbitalSimController
from orbitalengineer.ui import model
from orbitalengineer.ui.canvas import pz
from orbitalengineer.ui.gtk4 import GObject, Graphene


class Renderer(GObject.GObject):
    
    view = GObject.Property(type=object)
    data = GObject.Property(type=object)
    camera = GObject.Property(type=object)
    orbital:OrbitalSimController = GObject.Property(type=object)
    
    def __init__(self, view:model.ViewModel, data:model.DataModel, camera:pz.Camera2D, orbital:OrbitalSimController):
        super().__init__()
        self.view = view
        self.data = data
        self.camera = camera
        self.orbital = orbital
        
    def get_cairo(self, snapshot, width, height):
        return snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))
    
    def do_draw(self, snapshot, width, height):
        cr = self.get_cairo(snapshot, width, height)
        self.draw(cr, width, height)

    def draw(self, cr:cairo.Context, width:int, height:int):
        raise NotImplementedError()
