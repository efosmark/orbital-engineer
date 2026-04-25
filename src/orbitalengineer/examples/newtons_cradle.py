from orbitalengineer.engine.particle import Particle
from orbitalengineer.ui.mainapp import App


def on_activate(app: App):
    Lx = 4
    N = (Lx * 2)
    m = 2000
    r = 30
    v = 1

    # Just to c enter things a bit
    origin = 0 # (((N-1) * r) - (r*10))

    for i in range(N-1):
        app.insert_particle(Particle(
            position=complex(origin + (i * r * 2), 0),
            mass=m,
            radius=r,
            velocity=complex(-v/(N-1), 0)
        ), color=(1,1,1,1))

    app.insert_particle(Particle(
        position=complex(origin - (r*10), 0),
        mass=m,
        radius=r,
        velocity=complex(v, 0)
    ), color=(1,1,1,1))


    app.view.show_debug_info = False
    app.view.show_grid = False
    app.orbit_ctl.Lx = Lx
    app.orbit_ctl.speed = 50.0
    app.orbit_ctl.dt_base = 1/24.0
    app.orbit_ctl.coef_of_restitution = 0.99999
    app.orbit_ctl.init_sim()
    app.ticker.start()
    app.relative_zoom(1.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
