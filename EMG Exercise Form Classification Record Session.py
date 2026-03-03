# record_session.py
# Records a labeled EMG session from the Arduino over Serial
# Usage: python record_session.py
# --------------------------------------------------------

import serial
import csv
import time
import os
from datetime import datetime
from config import PORT, BAUD_RATE, OUTPUT_DIR, REP_THRESHOLD, REP_HOLD_MS, ZERO_CAL_SECONDS


def record_session(label: str, duration_seconds: int):
    """Record one labeled session of EMG data and save to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{OUTPUT_DIR}/{label}_{timestamp}.csv"

    print(f"\n>>> Recording '{label}' for {duration_seconds}s → {filename}")

    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(1)           # let serial connection stabilize
    ser.reset_input_buffer()

    # --- Zero-point calibration ---
    print(f"Relax your arm. Measuring zero point for {ZERO_CAL_SECONDS}s...")
    cal_samples = []
    cal_end = time.time() + ZERO_CAL_SECONDS
    while time.time() < cal_end:
        try:
            line = ser.readline().decode('utf-8').strip()
            if ',' not in line:
                continue
            _, value = line.split(',')
            cal_samples.append(int(value))
        except Exception:
            continue

    zero_point = round(sum(cal_samples) / len(cal_samples)) if cal_samples else 512
    print(f"Zero point: {zero_point} (nominal center = 512)\n")

    # --- Countdown ---
    print("Get into position... starting in 3 seconds.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("GO! Waiting for first rep...\n")

    # --- Wait for sustained activation above REP_THRESHOLD (RMS-based) ---
    # Raw EMG is AC — single-sample deviations are small even during contraction.
    # RMS over a short window captures the true signal envelope.
    RMS_WIN = 25            # samples (~50ms at 500Hz)
    rms_buf = []
    above_since = None
    hold_needed = REP_HOLD_MS / 1000.0
    arduino_ms_offset = 0
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if ',' not in line:
                continue
            arduino_ms_raw, value = line.split(',')
            value = int(value)
        except Exception:
            continue

        rms_buf.append(value - zero_point)
        if len(rms_buf) > RMS_WIN:
            rms_buf = rms_buf[-RMS_WIN:]
        if len(rms_buf) < RMS_WIN:
            continue

        rms = (sum(x * x for x in rms_buf) / RMS_WIN) ** 0.5

        now = time.time()
        if rms >= REP_THRESHOLD:
            if above_since is None:
                above_since = now
            elif (now - above_since) >= hold_needed:
                arduino_ms_offset = int(arduino_ms_raw)
                print(f"First rep detected (RMS={rms:.1f})! Recording started.\n")
                break
        else:
            above_since = None

    start = time.time()
    rows_written = 0

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arduino_ms', 'emg_raw', 'label'])

        while (time.time() - start) < duration_seconds:
            try:
                line = ser.readline().decode('utf-8').strip()
                if ',' in line:
                    arduino_ms, value = line.split(',')
                    writer.writerow([int(arduino_ms) - arduino_ms_offset, int(value) - zero_point, label])
                    rows_written += 1
            except Exception as e:
                print(f"Read error: {e}")
                continue

    ser.close()
    print(f"Done! Recorded {rows_written} samples → {filename}")
    return filename


def main():
    print("=== EMG Data Collection ===")
    print(f"Port: {PORT} | Baud: {BAUD_RATE}")
    print("Labels: 'good' = good form, 'bad' = bad form\n")

    while True:
        label = input("Enter label (good/bad) or 'quit': ").strip().lower()
        if label == 'quit':
            break
        if label not in ('good', 'bad'):
            print("Please enter 'good' or 'bad'")
            continue

        duration = input("Duration in seconds (e.g. 30): ").strip()
        if not duration.isdigit():
            print("Please enter a valid number")
            continue

        record_session(label, int(duration))

        cont = input("\nRecord another session? (y/n): ").strip().lower()
        if cont != 'y':
            break

    print(f"\nAll sessions saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()