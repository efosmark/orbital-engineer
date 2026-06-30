from orbitalengineer.engine.orbitalcl import flags
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, random_position, rng

import matplotlib.pyplot as plt
from matplotlib import colors

SOL_COLOR = (0.95, 0.8, 0.45, 1.0)


cmap = plt.colormaps['gist_rainbow']

Lx = 256
N = 1024
mass_min, mass_max = 1e2, 1e8
dist_min, dist_max = 2000, 200000

po = 7000
d = abs(po)

def on_activate(app: App):
    global dist_min, dist_max

    barycenter = create_primary(mass=1e12 / 4.0, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.FIXED_RADIUS)
    #barycenter._radius = 1    

    app.insert_particle(create_secondary(
        barycenter,
        mass=1e12,
        position=complex(0, -10000),
        flags=flags.MERGE,
    ), color=SOL_COLOR)
    
    app.insert_particle(create_secondary(
        barycenter,
        mass=1e12,
        position=complex(0, 10000),
        flags=flags.MERGE,
    ), color=SOL_COLOR)


    #sol = create_primary(mass=1e10, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.FIXED_RADIUS|flags.REPEL_ON_OVERLAP)
    #sol._radius = 30_000
    #app.insert_particle(sol, color=(1,1,1,0.05))

    #sol1 = create_primary(mass=3.5e10, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.FIXED_RADIUS|flags.MERGE_AS_PRIMARY)
    #sol1._radius = 50
    #app.insert_particle(sol1, color=SOL_COLOR)

    #dist_norm = colors.Normalize(sol1.get_radius() + dist_min, dist_max)

    barycenter = create_primary(mass=2e12, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.FIXED_RADIUS)
    
    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        app.insert_particle(create_secondary(
            barycenter,
            mass=mass,
            position=pos,
            #ecc=1.0,
            #radius=50,
            #prograde=(i <= N/2.0),
            #flags=flags.MERGE_AS_SECONDARY|flags.BOUNCE|flags.REPEL_ON_OVERLAP,
            flags=flags  .MERGE,
        #), color=cmap(1-dist_norm(abs(pos))))
        ), color=(1,1,1,1))

    app.orbital.Lx = Lx
    app.orbital.coef_of_restitution = 0.98
    app.view.show_focused_history = True
    app.view.show_debug_info = False
    app.view.show_focus_info = True
    app.relative_zoom(1/200.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
