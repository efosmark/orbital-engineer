import math

from orbitalengineer import flags
from orbitalengineer.engine.particle import Particle, ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, r_from_mass, random_color, random_position, rng

import matplotlib.pyplot as plt
cmap = plt.colormaps['gist_rainbow']

MASS = 1000
RADIUS = r_from_mass(MASS) #type:ignore

def populate(app: App):
    app.insert_particle(ParticleRaw(
        position=complex(0, -RADIUS),
        velocity=0,
        mass=MASS,
        radius=RADIUS,
        flags=flags.BOUNCE,
    ), color=(1, 0.5, 0.5, 1))

    app.insert_particle(ParticleRaw(
        position=complex(0, RADIUS),
        velocity=0,
        mass=MASS,
        radius=RADIUS,
        flags=flags.BOUNCE,
    ), color=(1, 0.5, 0.5, 1))

    # app.insert_particle(ParticleRaw(
    #     position=complex(RADIUS, 0),
    #     velocity=0,
    #     mass=MASS,
    #     radius=RADIUS,
    #     flags=flags.BOUNCE,
    # ), color=(1, 0.5, 0.5, 1))

    app.insert_particle(ParticleRaw(
        position=complex(-RADIUS * 5, 0),
        velocity=0+0j,
        mass=MASS * 2.0,
        radius=RADIUS,
        flags=flags.BOUNCE,
    ), color=(1, 1, 1, 1))

def on_activate(app: App):
    app.view.show_focused_history = True
    app.view.show_debug_info = False
    app.view.show_focus_info = True
    app.bootstrap()
    
    populate(app)
    
    #if not app.client.is_initialized:
    app.client.init_sim()
    
    #for i in range(78):
    #    app.tick_once()

def run():
    app = App(0, 0)
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
