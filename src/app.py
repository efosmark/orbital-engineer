from orbitalengineer import flags
from orbitalengineer.engine.particle import Particle, ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, r_from_mass, random_color, random_position, rng

import matplotlib.pyplot as plt
from matplotlib import colors
cmap = plt.colormaps['gist_rainbow']

def populate(app: App):
    for i in range(1000):
        app.insert_particle(ParticleRaw(
            position=random_position(0, 10000),
            velocity=random_position(0,   100),
            mass=100_000,
            radius=100,
            flags=flags.BOUNCE|flags.MERGE_AS_SECONDARY,
        ), color=(*random_color(), 1.0))

    # N = 1024
    # mass_min, mass_max = 100,  10_000_000
    # dist_min, dist_max = 20_000,   45_000
    # dist_norm = colors.Normalize(dist_min, dist_max)
    # center = create_primary(mass=1e11, flags=flags.MERGE_AS_PRIMARY)
    # app.insert_particle(center, color=(0.95, 0.8, 0.05, 1.0))
    # for i in range(N-1):
    #     mass = rng.uniform(mass_min, mass_max)
    #     pos = random_position(dist_min, dist_max)
    #     app.insert_particle(create_secondary(
    #         center,
    #         mass=mass,
    #         position=pos,
    #         flags=flags.BOUNCE|flags.MERGE_AS_SECONDARY,
    #     ), color=cmap(1-dist_norm(abs(pos))))
    # app.relative_zoom(1/100.0)


def on_activate(app: App):
    app.view.show_focused_history = True
    app.view.show_debug_info = False
    app.view.show_focus_info = True
    app.bootstrap()
    
    if not app.client.is_initialized:
        app.client.coef_of_restitution = 0.1
        populate(app)
        app.client.init_sim()

def run():
    app = App(0, 0)
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
