from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, random_position, rng

import matplotlib.pyplot as plt
from matplotlib import colors

SOL_COLOR = (0.95, 0.8, 0.05, 1.0)


cmap = plt.colormaps['gist_rainbow']

Lx = 256
N = (1024 * 4) - 2
mass_min, mass_max = 100, 10000
dist_min, dist_max = 1000, 10_000

po = 7000
d = abs(po)

def on_activate(app: App):
    global dist_min, dist_max

    center = create_primary(mass=1e8, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.FIXED_RADIUS)
    app.insert_particle(center, color=SOL_COLOR)

    app.insert_particle(create_secondary(
        center,
        mass=1e7,
        position=complex(0, dist_max / 2.0),
        flags=flags.MERGE,
    ), color=(1,1,1,1))

    dist_norm = colors.Normalize(dist_min, dist_max)
    
    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        app.insert_particle(create_secondary(
            center,
            mass=mass,
            position=pos,
            #ecc=1.0,
            #radius=50,
            #prograde=(i <= N/2.0),
            #flags=flags.REPEL_ON_OVERLAP,
            #flags=flags.MERGE_AS_SECONDARY|flags.BOUNCE|flags.REPEL_ON_OVERLAP,
            flags=flags.MERGE,
        ), color=cmap(1-dist_norm(abs(pos))))

    #app.orbital.Lx = Lx
    app.orbital.coef_of_restitution = 0.98
    app.view.show_focused_history = True
    app.view.show_debug_info = False
    app.view.show_focus_info = True
    app.relative_zoom(1/40.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
