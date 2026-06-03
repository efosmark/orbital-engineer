from orbitalengineer.engine.particle import ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import angular_position, r_from_mass, random_color, create_primary, create_secondary, random_position, rng

import time

N = 4

def on_activate(app: App):
    app.insert_particle(ParticleRaw(
            position=0+0j,
            velocity=0+0j,
            radius=100,
            mass=20
    ), color=(*random_color(), 1.0))
    
    app.insert_particle(ParticleRaw(
            position=-100+0j,
            velocity=0+0j,
            radius=100,
            mass=20
    ), color=(*random_color(), 1.0))
    
    app.insert_particle(ParticleRaw(
            position=100+0j,
            velocity=0+0j,
            radius=100,
            mass=20
    ), color=(*random_color(), 1.0))
    
    app.insert_particle(ParticleRaw(
            position=200+0j,
            velocity=0+0j,
            radius=100,
            mass=20
    ), color=(*random_color(), 1.0))
    

    app.orbital.Lx = N
    app.orbital.speed = 1.0
    app.orbital.coef_of_restitution = 0.999
    app.orbital.init_sim()
    app.orbital.tick(app.clock.time())
    time.sleep(app.orbital.dt_base)
    app.orbital.tick(app.clock.time())
    #app.ticker.start()
    #app.relative_zoom(1.0)

def run():
    app = App()
    on_activate(app)
    
    #app.connect("activate", on_activate)
    #app.run(None)

if __name__ == "__main__":
    run()
