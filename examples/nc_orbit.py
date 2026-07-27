from orbitalengineer import flags
from orbitalengineer.ui.mainapp import App
from orbitalengineer.helpers import create_primary, create_secondary


SOL_COLOR = (1.0, 0.98, 0.45, 1.0)


Lx = 1
N = (Lx  * 3) #- 1 
mass_min, mass_max = 100, 5000
vel_min, vel_max = 0, 0
radius_min, radius_max = 15, 100
dist_min, dist_max = 5_000, 15_000


def on_activate(app: App):
    global dist_min, dist_max
    sol = create_primary(mass=1e8, flags=flags.BOUNCE)
    app.insert_particle(sol, color=SOL_COLOR)
    sol._radius = 1000 # type:ignore

    mass = 5e6
    
    dist = 3_000
    radius = 200
    
    app.insert_particle(create_secondary(
        sol,
        mass=mass,
        radius=radius,
        position=complex(dist, 0),
        prograde=False,
        flags=flags.BOUNCE
    ), color=(1,1,1,1))
    
    app.insert_particle(create_secondary(
        sol,
        mass=mass,
        radius=radius,
        position=complex(-dist, 0),
        flags=flags.BOUNCE
    ), color=(1,1,1,1))

    app.client.coef_of_restitution = 0.999
    app.relative_zoom(1/30.0)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
