from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.particle import ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import random_position, rng

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = plt.colormaps['plasma']

Lx = 16
N = 768

mass_min, mass_max = 1e2, 1e6
dist_min, dist_max = 1, 9000

mass_norm = colors.Normalize(mass_min, mass_max)

def on_activate(app: App):
    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        velocity = 0+0j
        #radius = np.cbrt(mass / np.pi) / 5.0
        radius = 30.0
        
        app.insert_particle(ParticleRaw(
            position=pos,
            velocity=velocity,
            radius=radius,
            mass=mass,
            flags=flags.MERGE#|flags.DISABLE_BOUNCE|flags.REPEL_ON_OVERLAP
         ), color=cmap(mass_norm(mass)))

    # for i in range(128):
    #     pos = random_position(dist_min, dist_max)
    #     #velocity = random_posit ion(0, 20)
    #     mass = -mass_min
    #     velocity = 0+0j
    #     #radius = np.cbrt(abs(mass) / np.pi) / 5.0
        
    #     app.insert_particle(ParticleRaw(
    #         position=pos,
    #         velocity=velocity,
    #         radius=10.0,
    #         mass=mass
    #     ), color=(0.5,1,0.5,1))

    app.orbital.Lx = Lx
    
    #app.orbital.collision_strategy = CollisionStrategy.BOUNCE
    app.orbital.coef_of_restitution = 0.92
    app.view.show_focused_history = True
    #app.view.show_debug_info = False
    #app.view.show_focus_info = True
    #app.data.secondary_body = 41
    #app.view.show_plot_at_startup = False
    app.relative_zoom(1.0)
 
def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
