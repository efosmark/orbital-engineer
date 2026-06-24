from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import angular_position, random_color, create_primary, create_secondary, random_position, rng

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = plt.colormaps['gist_rainbow']

from orbitalengineer.engine import config
config.EPS_DIST = 0.00001
config.EPS_TIME = 0.00001

N = 512
Lx = 256

dist_min, dist_max = 0, 700
dist_norm = colors.Normalize(dist_min, dist_max)

def on_activate(app: App):
    sol = create_primary(mass=1e0)

    for i in range(N):
        position = random_position(dist_min, dist_max)
        app.insert_particle(create_secondary(
            sol,
            mass=100,
            position=position,
            flags=flags.BOUNCE
        ), color=cmap(1-dist_norm(abs(position))))
    
    app.orbital.Lx = Lx
    app.orbital.coef_of_restitution = 0.99
    #app.orbit_ctl.speed = 2.0
    
def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
