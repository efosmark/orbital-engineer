import math

from orbitalengineer import flags
from orbitalengineer.engine.particle import Particle, ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, r_from_mass, random_color, random_position, rng

import matplotlib.pyplot as plt
cmap = plt.colormaps['gist_rainbow']

def populate(app: App):
    # app.insert_particle(ParticleRaw(
    #     position=0,
    #     velocity=0,
    #     mass=100_000,
    #     radius=1,
    #     flags=flags.MERGE_AS_PRIMARY|flags.FIXED_RADIUS,
    # ), color=(*random_color(), 1.0))

    for i in range(512):
        mass = (1_000 * rng.random())
        #mass = (200_000 * rng.random()) + 100_000
        app.insert_particle(ParticleRaw(
            position=random_position(0, 850),
            velocity=random_position(0,   10),
            mass=mass,
            radius=math.cbrt(mass/math.pi),
            flags=flags.BOUNCE,
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
    app.model.show_focused_history = True
    app.model.show_debug_info = False
    app.model.show_focus_info = True
    app.bootstrap()
    
    populate(app)
    
    #if not app.client.is_initialized:
    app.client.init_sim()

def run():
    app = App(0, 0)
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
