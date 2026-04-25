
    # def draw_magnifier(
    #     self,
    #     cr:cairo.Context,
    #     idx,
    #     rx,
    #     ry,
    #     rwidth,
    #     rheight,
    #     scale=1.0
    # ):
    #     cr.set_source_rgb(0, 0, 0)
    #     cr.rectangle(rx, ry, rwidth, rheight)
    #     cr.fill()
        
    #     cr.set_source_rgb(1,1,1)
    #     cr.rectangle(rx, ry, rwidth, rheight)
    #     cr.stroke()
        
    #     cr.rectangle(rx, ry, rwidth, rheight)
    #     cr.clip()
        
    #     cr.save()
        
    #     b = self.orbital.body(idx)
        
    #     min_width = 3.0
    #     max_width = rwidth * 0.85
    #     diam = b.get_radius(self.step_id) * 2.0
    #     if diam > max_width:
    #         scale = (max_width / diam)
    #     elif diam < min_width:
    #         scale = (min_width / diam)
        
    #     cr.scale(scale, scale)
        
    #     # Move the body to the origin
    #     x, y = b.get_xy(self.step_id)
    #     cr.translate(-x, -y)
        
    #     # Offset the magnifier window origin
    #     cr.translate(rx/scale, ry/scale)
        
    #     # Center within the magnifier window
    #     cr.translate((rwidth/2/scale), (rheight/2/scale))

    #     self._draw_scene(
    #         cr, scale, 
    #         show_force_vectors=self.show_force_vectors,
    #         show_all_history=False, 
    #         show_history=False,
    #         show_orbital_ellipse=self.show_orbital_ellipse,
    #         show_reticle=False
    #     )
        
    #     cr.restore()