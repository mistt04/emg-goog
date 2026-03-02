# config.py
# Shared settings across all EMG scripts
# ----------------------------------------

# Serial port — run `ls /dev/cu.*` in terminal to find yours (Mac)
# On Windows it will look like 'COM3', 'COM4', etc.
PORT = '/dev/cu.usbmodem1401'
BAUD_RATE = 115200

# Data storage
OUTPUT_DIR = 'emg_sessions'

# Sampling
SAMPLE_RATE_HZ = 500  # must match Arduino sketch

# ML windowing
WINDOW_SIZE = 100       # samples per window (= 200ms at 500Hz)
WINDOW_OVERLAP = 0.5    # 50% overlap between windows
