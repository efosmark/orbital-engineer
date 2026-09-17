
APP_ID = "com.qmew.OrbitalEngineer"

WINDOW_DEFAULT_SIZE = (1000, 800)

DEFAULT_WINDOW_TITLE = "Orbital Engineer"

DEFAULT_SCENARIO_FILE = "/tmp/scenario.json"

DEFAULT_PARTICLE_COLOR_RGBA = (1, 1, 1, 1)

DEFAULT_FONT_FAMILY = "Monospace"
DEFAULT_FONT_SIZE  = 10

#############################################
# Audio Settings
#############################################


SAMPLE_RATE = 48000
BUFFER_SAMPLES = 1024
CHANNELS = 1
GST_CAPS = f"audio/x-raw,format=S16LE,layout=interleaved,channels={CHANNELS},rate={SAMPLE_RATE}"

AUDIO_PIPELINE_NAME = "src"
AUDIO_PIPELINE  = (
    f"appsrc name={AUDIO_PIPELINE_NAME} is-live=true format=time "
    "! audioconvert "
    "! audioresample "
    "! pipewiresink"
)