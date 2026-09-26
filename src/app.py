import math

from matplotlib import colors

from orbitalengineer import flags
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary, random_position, rng

import matplotlib.pyplot as plt
cmap = plt.colormaps['gist_rainbow']

SOL_COLOR = (1.0, 0.98, 0.45, 1.0)

def populate(app: App):
    sol = create_primary(
        mass=1e8,
        radius=200,
        flags=flags.FIXED_POSITION|flags.FIXED_RADIUS|flags.FIXED_VELOCITY|flags.MERGE_AS_PRIMARY
    )
    
    app.insert_particle(sol, color=SOL_COLOR)

    N = 4096
    mass_min, mass_max = 1_000, 500_000
    dist_min, dist_max =   500,  75_000
    dist_norm = colors.Normalize(dist_min, dist_max)
    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        app.insert_particle(create_secondary(
            sol,
            position=pos,
            mass=mass,
            #ecc=0.7,
            radius=math.cbrt(mass/math.pi),
            flags=flags.BOUNCE|flags.MERGE_AS_SECONDARY,
        ), color=cmap(1-dist_norm(abs(pos))))

def on_activate(app: App):
    app.bootstrap()
    populate(app)
    
    app.client.init_sim()
    app.model.show_focused_history = True
    app.model.show_debug_info = True
    app.model.show_focus_info = True
    app.model.secondary_body = 1011
    app.model.engine.clock_speed = 0.1
    app.show_message("Ready.", duration=10.0, desc="Press [space] to start.")

def run():
    app = App()
    app.connect("activate", on_activate)
    
    def on_body_merged(inst, obj):
        print(f"MERGE {obj.i} -> {obj.j}")
        return True
    
    app.connect("merging", on_body_merged)
    app.run(None)

if __name__ == "__main__":
    run()
