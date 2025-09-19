from orbitalengineer.ui.mainwindow import App
from orbitalengineer.helpers import random_color, create_primary, create_secondary, rng
import numpy as np
 
def on_activate(app: App):
    sol = create_primary(mass=5e5)
    app.canvas.add_element(sol, color=(1, 0.8, 0))
    app.canvas.primary_idx = 0
    
    N = 400
    r_min, r_max = 200, 10000
    for i in range(N):
        sec = create_secondary(
            sol,
            mass=rng.uniform(1, 1000),
            dist=(float(r_min), float(r_max))
        )
        app.canvas.add_element(sec, color=random_color())
        
    # N_DARK = 1
    # for i in range(N_DARK):
    #     color = (1,1,1)
            
    #     app.canvas.add_element(
    #         create_secondary(
    #             sol,
    #             dist=(r_min, r_max + 100),
    #             mass=1e2,
    #             radius=0,
    #         ),
    #         color=color
    #     )
    
    app.canvas.camera.zoom_at(0, 0, 0, 0, 0.2)
    app.start_simulation()

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
