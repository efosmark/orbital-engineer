#from orbitalengineer.engine.particle import Particle
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import angular_position, random_color, create_primary, create_secondary, random_position, rng

import matplotlib.pyplot as plt
from matplotlib import colors

SOL_COLOR = (1.0, 0.98, 0.45, 1.0)

def brighten(color:tuple[float,float,float,float], amount=0.1) -> tuple[float,float,float,float]:
    r = [*color]
    for i in range(len(color) - 1):
         r[i] = (color[i] + amount)# * color_norms[i]
    return (r[0], r[1], r[2], color[-1])

cmap = plt.colormaps['gist_rainbow']

Lx = 64
N = (Lx  * 4)
mass_min, mass_max = 1e2, 1e2
vel_min, vel_max = 0, 0
radius_min, radius_max = 20, 100
dist_min, dist_max = 100, 10000

po = 7000
d = abs(po)

def on_activate(app: App):
    global dist_min, dist_max
    sol = create_primary(mass=3e5)
    app.insert_particle(sol, color=SOL_COLOR)
    sol._radius = 400 # type: ignore

    dist_norm = colors.Normalize(sol.get_radius() + dist_min, dist_max)

    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)

        pos = random_position(sol.get_radius() + dist_min, dist_max)
        app.insert_particle(create_secondary(
            sol,
            mass=mass,
            radius=30,
            position=pos
        #), color=(1,1,1,1))
        ), color=brighten(cmap(1-dist_norm(abs(pos))), 0))
    
    # app.insert_particle(create_secondary(
    #     sol,
    #     mass=N * mass_max,
    #     #dist=dist,
    #     radius=200,
    #     #radius=r_from_mass(np.float64(mass))*2.5,
    #     #ecc=1.3
    #     position=(po) * -1
    # ), color=(1,1,1,1))

    app.orbit_ctl.Lx = Lx
    app.orbit_ctl.speed = 1.0
    app.orbit_ctl.coef_of_restitution = 0.99
    #app.view.show_debug_info = False
    #app.view.show_focus_info = True
    #app.data.secondary_body = 342
    #app.view.show_plot_at_startup = False
    app.orbit_ctl.init_sim()
    app.ticker.start()
    app.relative_zoom(1/10.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
