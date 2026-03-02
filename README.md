# EMG Exercise Form Classification

A machine learning project using the MyoWare 2.0 muscle sensor and Arduino Mega to classify whether a bicep curl is performed with good or bad form.

---

## File Structure

```
emg_project/
├── emg_sensor.ino          # Arduino sketch — upload to Mega once
├── config.py               # Shared settings — update PORT here
├── verify_signal.py        # Hardware/signal sanity check
├── record_session.py       # Records labeled EMG sessions to CSV
└── README.md               # This file
```

---

## Hardware

- MyoWare 2.0 Muscle Sensor
- Arduino Mega R3
- 3 gel EMG electrodes

**Wiring:**

| MyoWare Pin | Arduino Mega Pin |
|-------------|-----------------|
| `+`         | `5V`            |
| `-`         | `GND`           |
| `SIG`       | `A0`            |

**Electrode placement (bicep curl):**
- 2 recording electrodes along the bicep muscle belly, ~1-2cm apart, parallel to muscle fibers
- 1 reference electrode on a bony area (e.g. elbow)

---

## One-Time Setup

1. Open `emg_sensor.ino` in Arduino IDE
2. Select board: Tools → Board → Arduino Mega or Mega 2560
3. Find your port: Tools → Port (e.g. `COM3`)
4. Upload the sketch to the Mega (click the → button)
5. Update `PORT` in `config.py` to match your port
6. Install dependencies:
   ```
   pip install pyserial
   ```

---

## Recording Workflow

### Step 1 — Verify hardware
Run this first every session to confirm the sensor and wiring are working:
```
python verify_signal.py
```
Flex your muscle — the values and bar should visibly increase. Relax — they should drop back down. Runs for 10 seconds automatically. Press `Ctrl+C` to exit early.

### Step 2 — Record data
```
python record_session.py
```
You will be prompted:
- `Enter label (good/bad) or 'quit':` → type `good` or `bad`
- `Duration in seconds (e.g. 30):` → type `30`
- 3 second countdown → perform the exercise → CSV saves automatically
- `Record another session? (y/n):` → continue or quit

**Good form sessions:** strict controlled curl, full range of motion, no momentum

**Bad form sessions:** deliberate swinging, using shoulder/back momentum

**Target:** 5-6 sessions of each label per recording day

### Step 3 — Find your data
CSVs are saved automatically to the `emg_sessions/` folder:
```
emg_sessions/
├── good_20260302_143022.csv
├── good_20260302_143500.csv
├── bad_20260302_144001.csv
└── bad_20260302_144300.csv
```

---

## Consistency Rules
To ensure reliable data across sessions:
- Always use the **same weight**
- Always mark and use the **same electrode placement**
- Always use the **same arm**
- Maintain roughly the **same rep tempo**

---

## Notes
- `Ctrl+C` force quits any script at any time
- If the signal looks flat in `verify_signal.py`, check electrode contact and wiring
- The `emg_sessions/` folder is created automatically on first run
