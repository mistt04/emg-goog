# verify_signal.py
# Use this BEFORE recording to confirm your hardware is working correctly.
# Prints live EMG readings so you can verify muscle activation is being detected.
# Usage: python verify_signal.py
# -----------------------------------------------------------------------

import serial
import time
from config import PORT, BAUD_RATE


def verify(duration_seconds: int = 10):
    print(f"=== Signal Verification ({duration_seconds}s) ===")
    print(f"Connecting to {PORT} at {BAUD_RATE} baud...\n")

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Could not open port: {e}")
        print("Check that your Arduino is plugged in and PORT in config.py is correct.")
        print("Run `ls /dev/cu.*` (Mac) or check Device Manager (Windows) to find your port.")
        return

    time.sleep(1)
    ser.reset_input_buffer()

    print("Reading signal... flex and relax your muscle to test.")
    print(f"{'Time (s)':<12}{'EMG Raw':<12}{'Bar'}")
    print("-" * 50)

    start = time.time()
    while (time.time() - start) < duration_seconds:
        try:
            line = ser.readline().decode('utf-8').strip()
            if ',' in line:
                arduino_ms, value = line.split(',')
                value = int(value)
                elapsed = round(time.time() - start, 2)

                # Simple ASCII bar to visualize signal strength
                bar_length = int(value / 1023 * 40)
                bar = '█' * bar_length

                print(f"{elapsed:<12}{value:<12}{bar}")
        except Exception as e:
            print(f"Read error: {e}")
            continue

    ser.close()
    print("\nVerification complete.")
    print("If values jumped when you flexed → wiring and electrodes are good.")
    print("If values stayed flat or were all 0 → check wiring and electrode placement.")


if __name__ == '__main__':
    verify(duration_seconds=10)
