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
2. **RMS envelope** — smoothed activation signal computed from filtered EMG
3. **Rep segmentation** — adaptive per-session threshold (60th percentile of RMS) detects individual rep boundaries, minimum 150 sample duration. Yields 123 reps total (70 bad, 53 good)
4. **Feature extraction** per rep from the RMS envelope:
   - Time domain: RMS, MAV, waveform length, variance, skewness, kurtosis
   - Frequency domain: mean frequency
5. **Per-fold normalization** — normalize using training data statistics only, applied to test to prevent data leakage
6. **Random Forest classifier** (100 trees) trained and evaluated with leave-one-out cross-validation by session

## Results
Average accuracy: **79%**

| Session | Accuracy | Reps |
|---------|----------|------|
| bad_20260302_202831 | 0.79 | 14 |
| bad_20260302_203726 | 0.73 | 15 |
| bad_20260302_204144 | 0.92 | 12 |
| bad_20260302_204437 | 1.00 | 12 |
| bad_20260302_204556 | 1.00 | 17 |
| good_20260302_202655 | 0.67 | 12 |
| good_20260302_203636 | 0.80 | 10 |
| good_20260302_203807 | 0.77 | 13 |
| good_20260302_204220 | 0.82 | 11 |
| good_20260302_204517 | 0.43 | 7 |

## Key Findings
- Switching from fixed windows to rep-level segmentation was the biggest improvement (46% → 79%)
- Computing features on the RMS envelope instead of raw signal improved accuracy significantly (68% → 79%)
- **Waveform length is the top feature** — captures how smooth vs erratic the activation curve is over a rep, which directly reflects form quality
- `good_20260302_204517` is a persistent outlier (0.43) — likely due to fatigue. It was the last good session recorded and has the fewest reps (7). Fatigued good form produces a choppier signal that resembles bad form
- Bad sessions detect more reps (12-17) than good sessions (7-13) — consistent with bad form being physically easier/faster to perform

## Limitations
- Only 10 sessions (123 reps) — still a small dataset
- Single sensor on bicep only — bad form shifts activation to shoulder/back which isn't captured
- Fatigue confounds the signal — fatigued good form looks like bad form to the model
- Session-to-session variability from sensor placement remains a factor

## Recommendations for Next Steps
1. **More data** — aim for 30-50 sessions, mark exact sensor placement each time
2. **Additional sensors** — add one on the shoulder (deltoid) or trap to capture compensatory muscles
3. **Fatigue as a third class** — label fatigued good form separately (`fatigued_good`) to distinguish form breakdown from bad form. Could enable "time to rest" feedback in real use
4. **More exaggerated bad form** — make bad form very consistent and deliberate across sessions
