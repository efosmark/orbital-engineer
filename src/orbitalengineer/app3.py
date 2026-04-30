import time
from orbitalengineer.engine.particle import ParticleRaw
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import angular_position, r_from_mass, random_color, create_primary, create_secondary, random_position, rng

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
    

    app.orbit_ctl.Lx = N
    app.orbit_ctl.speed = 1.0
    app.orbit_ctl.coef_of_restitution = 0.999
    app.orbit_ctl.init_sim()
    app.orbit_ctl.tick(time.monotonic())
    time.sleep(app.orbit_ctl.dt_base)
    app.orbit_ctl.tick(time.monotonic())
    #app.ticker.start()
    #app.relative_zoom(1.0)

def run():
    app = App()
    on_activate(app)
    
    #app.connect("activate", on_activate)
    #app.run(None)

if __name__ == "__main__":
    run()
