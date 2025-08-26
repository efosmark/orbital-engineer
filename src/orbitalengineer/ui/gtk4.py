import gi
try:
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
except ValueError:
    print("GTK 4.0 not available")
    exit(1)
from gi.repository import Gtk, Gdk, Gsk, GLib, Gio, Adw, Graphene  # type: ignore
