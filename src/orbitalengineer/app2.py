from orbitalengineer.engine.particle import ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import angular_position, r_from_mass, random_color, create_primary, create_secondary, random_position, rng

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = plt.colormaps['plasma']

Lx = 64
N = (Lx  * 1)

mass_min, mass_max = 5e3, 5e6
dist_min, dist_max = 1, 3000

mass_norm = colors.Normalize(mass_min, mass_max)

def on_activate(app: App):
    for i in range(N):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        #velocity = random_position(0, 20)
        velocity = 0+0j
        radius = np.sqrt(mass / np.pi) / 5.0
        #radius = 10.0
        
        app.insert_particle(ParticleRaw(
            position=pos,
            velocity=velocity,
            radius=radius,
            mass=mass
        ), color=(*random_color(), 1.0))
        #), color=cmap(mass_norm(mass)))

    app.orbit_ctl.Lx = Lx
    app.orbit_ctl.speed = 1.0
    app.orbit_ctl.coef_of_restitution = 0.999
    app.view.show_focused_history = True
    #app.view.show_debug_info = False
    #app.view.show_focus_info = True
    app.data.secondary_body = 41
    #app.view.show_plot_at_startup = False
    app.orbit_ctl.init_sim()
    app.ticker.start()
    app.relative_zoom(1.0)
 
def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
