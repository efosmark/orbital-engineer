from orbitalengineer.ui.mainapp import App

def on_activate(app: App):
    app.bootstrap(reset=False)

def run():
    app = App()
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    run()
