# Flatpak development build

The manifest targets GNOME 49 (Python 3.13) on x86_64. Install the runtime and SDK before building:

```sh
flatpak install flathub org.gnome.Platform//49 org.gnome.Sdk//49
make dev
make flatpak-check
make flatpak-build
make flatpak-install
```

`flatpak-build` builds locally without installing the app. `flatpak-install`
also installs it for the current user. Build output and builder state stay in this directory and are ignored by Git. Wheel downloads include the engine requirements from the repository root plus Matplotlib; GNOME supplies PyGObject, Cairo, GTK 4, libadwaita, and GStreamer. The application wheel includes the OpenCL kernel sources.

The server and UI are separate processes. Start each in its own terminal:

```sh
make flatpak-server
make flatpak-run
```

The metrics UI can run in a third terminal:

```sh
flatpak run --command=python3 com.qmew.OrbitalEngineer -m ui_metrics.metrics_app
```

The manifest shares the network namespace for the TCP connection between the UI and server, IPC for shared memory, and the audio socket for GStreamer.

The server currently binds to all network interfaces on TCP port 50055; use these development builds only on a trusted network.

GPU execution requires an OpenCL driver and ICD visible inside the Flatpak sandbox. Installing PyOpenCL wheels and granting DRI access does not install the host's ROCm driver inside the sandbox. Verify driver availability with:

```sh
flatpak run --command=python3 com.qmew.OrbitalEngineer -c \
  'import pyopencl as cl; print([(p.name, [d.name for d in p.get_devices()]) for p in cl.get_platforms()])'
```

A missing platform or empty device list requires a compatible Flatpak OpenCL driver setup before simulation will work. The app also currently assumes an AMD device when reading GPU telemetry.

For another architecture or SDK, update the manifest and pass matching `PYVER_TAG` and `PLAT_TAG` values to `make vendor-wheels`. The defaults accept manylinux 2.17, 2.27, and 2.28 x86_64 wheels. Downloads use a fresh staging directory on every run so artifacts from different Python ABIs cannot mix.
