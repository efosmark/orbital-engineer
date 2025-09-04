from orbitalengineer.ui.mainwindow import App
from orbitalengineer.helpers import random_color, create_primary, create_secondary, rng
import numpy as np
 
def on_activate(app: App):
    sol = create_primary(mass=1e5)
    app.canvas.add_element(sol, color=(1, 0.8, 0))
    app.canvas.primary_idx = 0
    
    N = 5000
    r_min, r_max = 100, 1000
    
    u = rng.uniform(r_min*r_min, r_max*r_max, size=N)
    for r_b in np.sqrt(u):
        secondary = create_secondary(
            sol,
            mass=rng.uniform(1, 5),
            min_radius=float(r_b),
            max_radius=float(r_b)
        )
        app.canvas.add_element(
            secondary,
            color=random_color()
        )
    
    app.canvas.secondary_idx = 215
    app.canvas.camera.zoom_at(0, 0, 0, 0, 0.99)
    app.start_simulation()

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
