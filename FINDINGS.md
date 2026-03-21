# EMG Form Classification — ML Findings
**Date:** 2026-03-21

## Goal
Use Arduino EMG readings from the bicep to classify good vs bad form on a single-arm bicep curl using a machine learning model.

## Data
- 10 sessions total (5 good, 5 bad) recorded at 500Hz
- Good form: strict curl, no swinging, isolated bicep
- Bad form: swinging motion, offloading to shoulder and back
- EMG sensor placed on muscle belly of bicep, grounded at elbow
- Each session ~20 seconds of continuous recording, baseline-corrected and rep-detection triggered before recording starts

## Pipeline Built (`ml/`)
1. **Bandpass filter** (20-240Hz) to remove motion artifacts and noise
2. **Windowing** — 100 sample windows (200ms) with 50% overlap → ~195 windows per session, 1950 total
3. **Feature extraction** per window:
   - Time domain: RMS, MAV, zero crossing rate, waveform length, variance, skewness, kurtosis
   - Frequency domain: mean frequency, median frequency
4. **Per-session normalization** to reduce inter-session variability
5. **Random Forest classifier** (100 trees) trained and evaluated with leave-one-out cross-validation

## Results
Average accuracy: **52%** (essentially random, chance = 50%)

| Session | Accuracy |
|---------|----------|
| bad_20260302_202831 | 0.11 |
| bad_20260302_203726 | 0.63 |
| bad_20260302_204144 | 0.30 |
| bad_20260302_204437 | 0.76 |
| bad_20260302_204556 | 0.81 |
| good_20260302_202655 | 0.68 |
| good_20260302_203636 | 0.69 |
| good_20260302_203807 | 0.62 |
| good_20260302_204220 | 0.54 |
| good_20260302_204517 | 0.05 |

## Key Findings
- Feature averages DO differ between good and bad (good form shows ~38% higher RMS/MAV — consistent with bad form offloading work away from the bicep)
- However, variance within sessions is too high for the model to generalize across sessions
- Some sessions classify well (0.81), others perform worse than random (0.05, 0.11) — indicating inconsistent signal characteristics across recording days
- Model is not the bottleneck — a single bicep sensor likely doesn't capture enough information to reliably distinguish good from bad form

## Limitations
- Only 10 sessions — too few for robust generalization
- Single sensor on bicep only — bad form shifts activation to shoulder/back which isn't captured
- Session-to-session variability (sensor placement, degree of bad form) is too high

## Recommendations for Next Steps
1. **More data** — aim for 30-50 sessions, mark exact sensor placement each time
2. **Additional sensors** — add one on the shoulder (deltoid) or trap to capture the compensatory muscles used in bad form
3. **More exaggerated bad form** — make bad form very consistent and deliberate across sessions
