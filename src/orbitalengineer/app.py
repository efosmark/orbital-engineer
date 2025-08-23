import random

from orbitalengineer.ui.mainwindow import App
from orbitalengineer.helpers import random_color, create_primary, create_secondary

import numpy as np
rng = np.random.default_rng()
 
def on_activate(app: App):
    sol = create_primary(mass=1e6)
    app.canvas.add_element(sol, color=(1, 0.8, 0))
    app.canvas.primary_idx = 0
    
    # AU = 149_597_871 / sol.radius
    
    # planets = [
    #     #    Dist     Mass      Color
    #     (    0.39,    0.055,    (0.9, 0.9, 0.2, 1)   ),   # Mercury
    #     (    0.72,    0.815,    (0.9, 0.9, 0.2, 1)   ),   # Venus
    #     (    1.0,     1.0,      (0.1, 0.1, 1.0, 1)   ),   # Earth
    #     (    1.52,    0.107,    (1.0, 0.1, 0.1, 1)   ),   # Mars
    #     (    5.2,     317.8,    (1.0, 0.7, 0.1, 1)   ),   # Jupiter
    #     (    9.5,     95.2,     (1.0, 1.0, 0.4, 1)   ),   # Saturn
    #     (    19.2,    14.5,     (1.0, 0.7, 0.7, 1)   ),   # Uranus
    #     (    30.1,    17.1,     (1.0, 0.1, 0.7, 1)   ),   # Neptune
    # ]
    
    # for p in planets:
    #     body = create_secondary(
    #         sol,
    #         mass=p[1]*1000,  
    #         min_radius=AU * p[0],
    #         ma x_radius=AU * p[0]
    #     )
    #     app.canvas.add_element(body, color=p[2])
    
    N = 200
    r_min, r_max = 3000, 4000
    
    u = rng.uniform(r_min*r_min, r_max*r_max, size=N)
    for r_b in np.sqrt(u):
        secondary = create_secondary(
            sol,
            mass=random.uniform(1, 1000),
            min_radius=float(r_b),
            max_radius=float(r_b)
        )
        
        app.canvas.add_element(
            secondary,
            color=random_color()
        )
        
    app.canvas.secondary_idx = random.choice([b for b in app.canvas.body_meta.keys() if b != app.canvas.primary_idx])
    app.canvas.show_history = False
    app.canvas.show_map = True
    app.canvas.show_force_vectors = False
    app.canvas.track_focused = False 
    app.canvas.camera.zoom_at(0, 0, 0, 0, 0.05)
    
    #app.start_maximized = True
    #app.canvas.add_tick_callback(app.on_tick)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)


if __name__ == "__main__":
    run()
