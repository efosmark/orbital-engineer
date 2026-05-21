from orbitalengineer.engine.cl import flags
from orbitalengineer.engine.collision_strategy import CollisionStrategy
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, random_position, rng

import matplotlib.pyplot as plt
from matplotlib import colors

SOL_COLOR = (1.0, 0.98, 0.45, 1.0)

def brighten(color:tuple[float,float,float,float], amount=0.1) -> tuple[float,float,float,float]:
    r = [*color]
    for i in range(len(color) - 1):
         r[i] = (color[i] + amount)# * color_norms[i]
    return (r[0], r[1], r[2], color[-1])

cmap = plt.colormaps['gist_rainbow']

Lx = 256
N = (Lx  * 2)
mass_min, mass_max = 1e2, 1e9
vel_min, vel_max = 0, 0
radius_min, radius_max = 20, 100
dist_min, dist_max = 1000, 120000

po = 7000
d = abs(po)

def on_activate(app: App):
    global dist_min, dist_max
    sol = create_primary(mass=3e10, flags=flags.FIXED_POSITION|flags.FIXED_VELOCITY|flags.BOUNCE_AS_PRIMARY|flags.MERGE_AS_PRIMARY)
    app.insert_particle(sol, color=SOL_COLOR)
    #sol._radius = 400 # type: ignore

    dist_norm = colors.Normalize(sol.get_radius() + dist_min, dist_max)

    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)

        pos = random_position(sol.get_radius() + dist_min, dist_max)
        app.insert_particle(create_secondary(
            sol,
            mass=mass,
            position=pos,
            ecc=0.99,
            flags=flags.BOUNCE|flags.MERGE_AS_SECONDARY,
        ), color=brighten(cmap(1-dist_norm(abs(pos))), 0))

    app.orbit_ctl.Lx = Lx
    app.orbit_ctl.speed = 1.0
    app.orbit_ctl.coef_of_restitution = 0.93
    app.view.show_focused_history = True
    #app.view.show_debug_info = False
    app.view.show_focus_info = True
    #app.data.secondary_body = 0
    app.orbit_ctl.init_sim()
    app.start_tick()
    app.relative_zoom(1/20.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
