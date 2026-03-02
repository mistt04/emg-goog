# record_session.py
# Records a labeled EMG session from the Arduino over Serial
# Usage: python record_session.py
# --------------------------------------------------------

import serial
import csv
import time
import os
from datetime import datetime
from config import PORT, BAUD_RATE, OUTPUT_DIR


def record_session(label: str, duration_seconds: int):
    """Record one labeled session of EMG data and save to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{OUTPUT_DIR}/{label}_{timestamp}.csv"

    print(f"\n>>> Recording '{label}' for {duration_seconds}s → {filename}")
    print("Get into position... starting in 3 seconds.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("GO!\n")

    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(1)           # let serial connection stabilize
    ser.reset_input_buffer() # clear any buffered junk

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
                    writer.writerow([arduino_ms, value, label])
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
