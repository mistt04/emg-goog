# CLAUDE.md — EMG Form Classification Project

## Project Overview
Arduino-based EMG system that records bicep muscle activation during exercise and uses ML to classify good vs bad form in real time.

## Exercise Recorded So Far
- **Single-arm bicep curl** (right arm)
- **Good form:** strict isolated curl, no elbow movement, no swinging
- **Bad form:** swinging motion, offloading work to shoulder and back, less bicep engagement

## Hardware & Recording
- EMG sensor placed on **muscle belly of the bicep**, grounded at the elbow
- Arduino samples at **500Hz**
- Sessions are ~20 seconds of continuous recording
- Zero-point calibration done at start of each session (2s rest baseline)
- Recording starts only after first rep is detected (RMS threshold = 50 ADC units, held for 150ms)
- Data saved as CSV: `arduino_ms, emg_raw, label`
- Config in `config.py` — port, baud rate, sample rate, thresholds all live here

## Data
- Located in `emg_sessions/` — 10 sessions total (5 good, 5 bad)
- All recorded 2026-03-02
- ~9800 samples per session (~19.7s at 500Hz)
- Values are baseline-corrected (zero_point subtracted) so signal oscillates around 0

## Project Structure
```
emg-goog/
├── config.py                                        # shared settings
├── EMG Sensor Code.ino                              # Arduino sketch
├── EMG Exercise Form Classification Record Session.py  # data collection script
├── EMG Exercise Form Classification Signal Verification.py  # signal check tool
├── emg_sessions/                                    # raw CSV data
└── ml/
    ├── preprocess.py                                # pipeline: filter, envelope, segmentation, features
    └── train.py                                     # cross-validation and model training
```

## ML Pipeline (current best: 79% accuracy)
1. **Bandpass filter** (20-240Hz) — removes motion artifacts and noise. 240Hz cap due to Nyquist at 500Hz sample rate
2. **RMS envelope** — 25-sample rolling RMS window smooths the signal into an activation curve
3. **Rep segmentation** — adaptive threshold per session (60th percentile of RMS), min 150 samples per rep (~0.3s). Yields 123 reps across 10 sessions
4. **Feature extraction** on the RMS envelope per rep:
   - Waveform length (WL) — top feature, captures smoothness of activation curve
   - Mean frequency
   - Variance
   - MAV (mean absolute value)
   - RMS
   - Kurtosis
   - Skewness
5. **Normalization** — per cross-validation fold using training stats only (no data leakage)
6. **Random Forest** — 100 trees, random_state=42, leave-one-out cross-validation by session

## Key Design Decisions
- Features computed on **RMS envelope**, not raw signal — envelope captures activation shape, raw signal captures electrical noise
- **Rep-level classification** not window-level — classifying individual curls is the actual goal
- **Leave-one-out CV by session** — prevents data leakage between windows from the same session
- ZC (zero crossing rate) and median frequency dropped — useless on the always-positive envelope

## Known Issues & Limitations
- `good_20260302_204517` is a persistent outlier (0.43 accuracy, 7 reps) — likely due to fatigue, was the last session recorded
- Bad sessions detect more reps (12-17) than good sessions (7-13) — bad form is faster to perform
- Single bicep sensor only — bad form shifts activation to shoulder/back which isn't captured
- Fatigue confounds the signal — fatigued good form resembles bad form

## Planned Next Steps
1. Collect more data (30-50 sessions), consistent sensor placement
2. Add shoulder/deltoid sensor to capture compensatory muscles
3. Add `fatigued_good` as a third label for fatigue detection ("time to rest" feedback)
4. More exaggerated and consistent bad form across sessions

## Running the ML Pipeline
```bash
cd ml/
python preprocess.py   # verify pipeline, prints X/y shapes
python train.py        # runs cross-validation, prints per-session accuracy
```

## See Also
- `FINDINGS.md` — full results and analysis
