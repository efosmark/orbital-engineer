# UI Front-End

A UI front-end for the Orbital Engineer engine.

## TODO

### Features

- Create a separate OrbitalApp that extends Gtk.Application and exposes all of the necessary engine components & events
- Add a zoom slider
- Move the tick count to the hud_clock component

### Optimizations

- Ellipse: ensure get_focus and get_orbit_info are only called when the followed body changes
- Focus Info: Do not recompute the info unless the followed body changes
- GravField: Expensive. Do not recompute constantly. Consider GPU acceleration.
- History Renderer: Shouldn't be so expensive. Do not compute in the rendering section.
- Remove usage of Renderer.orbital -- Prefer Renderer.app.engine.
- Renderer.camera.zoom is the only property used. Consider moving to a Gtk Property.

### Cleanup

- Debug Info: Remove (ensure the info is available elsewhere)
- Event Ledger: figure out if this should be removed or turned into something else. Perhaps show the raw events + the composite events
- Force Renderer: Verify this is working. Remove if it isn't.
- Grid Renderer: Messy, but works. Clean up code and add documentation.
- HUD Clock: Move inline style settings to constants at the top of the file.
- Magnifier: Get it working or just remove it.
- Pinpoint Renderer: Get it working or remove it.
- Warning renderer: Remove it, but put the warning in the HUD clock component
- MoveParticleController is over-complicated. Simplify and verify for mouse. Ensure touch screen support does not break.
