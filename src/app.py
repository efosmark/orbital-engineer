import math

from matplotlib import colors

from orbitalengineer import flags
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.helpers import create_primary, create_secondary, random_position, rng
from gui.mainapp import App

import matplotlib.pyplot as plt
cmap = plt.colormaps['gist_rainbow']

SOL_COLOR = (1.0, 0.98, 0.45, 1.0)

def populate(app: App):
    N = 1024
    mass_min, mass_max =  1_000,   5_000
    dist_min, dist_max =  5_000, 105_000
    
    sol = create_primary(mass=1e9, radius=300)
    app.insert_particle(sol, color=SOL_COLOR)
    
    dist_norm = colors.Normalize(dist_min, dist_max)
    for i in range(N-1):
        mass = rng.uniform(mass_min, mass_max)
        pos = random_position(dist_min, dist_max)
        app.insert_particle(create_secondary(
            sol,
            position=pos,
            mass=mass,
            radius=math.sqrt(mass/math.pi),
            #flags=flags.BOUNCE
        ), color=cmap(1-dist_norm(abs(pos))))

def on_activate(app: App):
    app.bootstrap()
    populate(app)
    
    app.client.init_sim(SimConfig(
        DEFAULT_DT_BASE=1/10.0,
        TARGET_TICKS_PER_SECOND=120,
        SOFTENING_LENGTH=100.0,
        COEF_OF_RESTITUTION=1.0
    ))
    app.model.show_focused_history = True
    app.model.show_debug_info = True
    app.model.show_focus_info = True
    app.relative_zoom(1/50.0)
    app.show_message("Ready.", duration=10.0, desc="Press [space] to start.")

def run():
    app = App(0, 0)
    app.connect("activate", on_activate)
    
    def on_body_merged(inst, obj):
        print(f"MERGE {obj.i} -> {obj.j}")
        return True
    
    app.connect("merging", on_body_merged)
    
    def on_tick(inst, tick_id):
       if app.model.engine.clock_time >= 120:
           app.model.engine.paused = True
    app.connect("tick", on_tick)
    
    app.run(None)

if __name__ == "__main__":
    run()
