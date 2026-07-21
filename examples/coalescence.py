from orbitalengineer import flags
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, random_position

import matplotlib.pyplot as plt
from matplotlib import colors

cmap = plt.colormaps['gist_rainbow']

from orbitalengineer.engine import config
config.EPS_DIST = 0.0001
config.EPS_TIME = 0.0001

N = 2048
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

    app.orbital.coef_of_restitution = 0.9
    
def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
