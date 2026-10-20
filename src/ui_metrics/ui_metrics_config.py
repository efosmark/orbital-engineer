APP_ID = "com.qmew.orbitalengineer.metrics"

WINDOW_TITLE = "Metrics Monitor"

WINDOW_DEFAULT_SIZE = (1200, 1000)

MAX_METRICS_PER_POLL = 500

REFRESH_INTERVAL_MS = 250

# Retain all samples within this many seconds of the latest simulation time.
UTILIZATION_HISTORY_SECONDS = 60
GPU_TEMPERATURE_HISTORY_SECONDS = 60
ENERGY_HISTORY_SECONDS = 180
DURATION_HISTORY_SECONDS = 60
DT_STEP_HISTORY_SECONDS = 60
VIRIAL_RATIO_HISTORY_SECONDS = 180

# Fallback until enough nonzero energy-error data is available to set the scale.
ENERGY_ERROR_ZERO_RANGE = 0.001

# Scale to the current history's 95th percentile of error magnitudes.
ENERGY_ERROR_SCALE_QUANTILE = 0.999
ENERGY_ERROR_SCALE_MIN_SAMPLES = 5
