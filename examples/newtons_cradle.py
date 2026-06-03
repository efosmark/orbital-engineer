from orbitalengineer.engine.cl import flags
from orbitalengineer.engine.particle import ParticleRaw
from orbitalengineer.helpers import random_color
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
        app.insert_particle(ParticleRaw(
            position=complex(origin + (i * r * 2), 0),
            mass=m,
            radius=r,
            velocity=complex(-v/(N-1), 0),
            flags=flags.BOUNCE
    ), color=(*random_color(), 1.0))

    app.insert_particle(ParticleRaw(
        position=complex(origin - (r*10), 0),
        mass=m,
        radius=r,
        velocity=complex(v, 0),
        flags=flags.BOUNCE
    ), color=(*random_color(), 1.0))

    app.orbital.Lx = Lx
    app.orbital.coef_of_restitution = 0.99999

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
