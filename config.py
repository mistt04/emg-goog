# config.py
# Shared settings across all EMG scripts
# ----------------------------------------

# Serial port — run `ls /dev/cu.*` in terminal to find yours (Mac)
# On Windows it will look like 'COM3', 'COM4', etc.
PORT = 'COM5'
BAUD_RATE = 115200

# Data storage
OUTPUT_DIR = 'emg_sessions'

# Sampling
SAMPLE_RATE_HZ = 500  # must match Arduino sketch

# ML windowing
WINDOW_SIZE = 100       # samples per window (= 200ms at 500Hz)
WINDOW_OVERLAP = 0.5    # 50% overlap between windows

# Zero-point calibration
ZERO_CAL_SECONDS = 2       # seconds of rest data collected to measure the sensor's resting baseline

# First-rep auto-detection
# REP_THRESHOLD is compared against the RMS amplitude of a short rolling window
# (not a single sample). Raw EMG is AC, so instantaneous values are tiny even
# during a strong contraction — RMS captures the true envelope.
REP_THRESHOLD = 25         # RMS ADC units above zero; raise if false-triggers, lower if reps are missed
REP_HOLD_MS = 150          # ms the RMS must stay above threshold before recording starts
