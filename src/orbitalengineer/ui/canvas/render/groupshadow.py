from collections import defaultdict
import math
import cairo
from orbitalengineer.engine.orbitalnp.integrator.merge import center_of_mass_position, r_from_mass
from orbitalengineer.ui.canvas import renderer
import numpy as np

MIN_RADIUS = 0.5
DEFAULT_PARTICLE_COLOR = (1, 1, 1)

def draw_shapely(cr, geom):
    if geom.geom_type == "Polygon":
        draw_polygon(cr, geom)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            draw_polygon(cr, poly)

def draw_polygon(cr, poly):
    # Exterior boundary
    coords = list(poly.exterior.coords)
    cr.move_to(*coords[0])
    for x, y in coords[1:]:
        cr.line_to(x, y)
    cr.close_path()
    

class GroupShadowRenderer(renderer.Renderer):

    def draw(self, cr:cairo.Context, width:int, height:int):
        buffer_id = self.orbital.buffer_static
        
        shm_group = self.orbital.shm.group[buffer_id]
        
        
        groups = defaultdict(list)
        for body_id, group_id in enumerate(shm_group):
            groups[group_id].append(body_id)
        
        for group_id, body_ids in groups.items():
            if len(body_ids) <= 2 or group_id == -1:
                continue
            
            body_ids = np.array(body_ids, dtype=np.int64)
            r_cm = center_of_mass_position(
                body_ids,
                self.orbital.shm.mass[buffer_id],
                self.orbital.shm.position[buffer_id]
            )
            total_mass = np.sum(self.orbital.shm.mass[buffer_id, body_ids])
            radius = r_from_mass(total_mass) * 2.5
            r,g,b = self.view.particle_colors[group_id]
            cr.set_source_rgba(r,g,b,0.2)
            cr.arc(r_cm.real, r_cm.imag, radius, 0, 2*math.pi)
            cr.fill()
            # circles: [(x, y, r), ...]
            
            # circles = []
            # for b_id in body_ids:
            #     b = self.orbital.body(b_id)
            #     pos = b.get_position()
            #     r = b.get_radius() + 3.0
            #     circles.append((pos.real, pos.imag, r))
            # #circles = [(100, 100, 50), (140, 100, 50), (120, 140, 50)]

            # # compute geometric union
            # shapes = [
            #     Point(x, y).buffer(r) for x, y, r in circles]
            # blob = unary_union(shapes)

            # # Now draw the union boundary with cairo
            # #surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 400)
            # #cr = cairo.Context(surf)

            # cr.set_line_width(3/self.camera.zoom)

            # draw_shapely(cr, blob)
            # cr.close_path()

            # cr.stroke()