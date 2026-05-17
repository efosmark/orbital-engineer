from orbitalengineer.ui.canvas import renderer


BG_COLOR = (0.05, 0, 0.05, 0.7)
TEXT_COLOR = (0.7, 0.7, 0.7)
BORDER_COLOR = (0.1, 0.1, 0.1)


class OSDRenderer(renderer.Renderer):

    def draw(self, cr, width:int, height:int):
        if len(self.view.osd_message) == 0: return
                
        now = self.orbital.clock.time()
        existing_osd_message = []
        for m in self.view.osd_message:
            if 0 < m.until < now:
                continue
            
            cr.save()
            
            total_duration = m.until - m.start
            duration = now - m.start

            intensity = (1 - (duration/total_duration))
            
            te = cr.text_extents(m.message)
            cr.move_to(width/2 - te.width/2, height/2 - te.height)
            cr.select_font_face(self.view.font_family)
            cr.set_font_size(20)
            cr.set_source_rgba(1, 1, 1, intensity)
            cr.show_text(m.message)
            
            existing_osd_message.append(m)
        
            cr.restore()
        
        self.view.osd_message = existing_osd_message
