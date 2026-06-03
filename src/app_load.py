from orbitalengineer.ui.mainapp import App

def on_activate(app: App):
    ...
    #app.load_from_file()
    #serde.load_scenario(ui_config.DEFAULT_SCENARIO_FILE, app.orbital, app.view)
    #app.relative_zoom(1.0)
 
def run():
    app = App(resume_from_file=True, platform_id=0, device_id=0)
    app .connect("activate", on_activate)
    app.run(None)
 
if __name__ == "__main__":
    run()
