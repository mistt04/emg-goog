import pandas as pd
import numpy as np 
from scipy.signal import butter, filtfilt
from pathlib import Path 
import os

def bandpass_filter(signal, lowcut=20, highcut=240, fs=500):
    nyq = fs / 2 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')

    return filtfilt(b, a, signal)

def rms_envelope(signal, win=25):
    return np.array([np.sqrt(np.mean(signal[i:i+win]**2)) for i in range(len(signal))])

def load_sessions(data_dir='emg_sessions'):
    dfs = []
    for file in Path(data_dir).glob('*.csv'):
        df = pd.read_csv(file)
        df['session'] = file.stem
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    df = df.copy()
    df['emg_filtered'] = bandpass_filter(df['emg_raw'].values)
    df['emg_envelope'] = rms_envelope(df['emg_filtered'].values)
    return df

def create_windows(df, window_size=100, overlap=0.5):
    step = int(window_size * (1 - overlap))
    windows = []
    labels = []
    sessions = []

    for session, group in df.groupby('session'):
        signal = group['emg_filtered'].values
        label = group['label'].iloc[0]

        for start in range(0, len(signal) - window_size, step):
            window = signal[start:start + window_size]
            windows.append(window)
            labels.append(label)
            sessions.append(session)

    return np.array(windows), np.array(labels), np.array(sessions)

def extract_features(window, fs=500):
    # time-domain
    mav = np.mean(np.abs(window))
    wl = np.sum(np.abs(np.diff(window)))
    var = np.var(window)
    skew = pd.Series(window).skew()

    # freq with Hann window (fixes spectral leakage, guards against zero-sum)
    freqs = np.fft.rfftfreq(len(window), d=1/fs)
    hann = np.hanning(len(window))
    power = np.abs(np.fft.rfft(window * hann))**2
    mean_freq = np.sum(freqs * power) / (np.sum(power) + 1e-8)

    # envelope shape features
    peak_idx = np.argmax(window)
    peak_loc = peak_idx / max(len(window) - 1, 1)
    # where in the rep the peak occurs (0=start, 1=end); good form peaks ~0.4-0.6

    peak_ratio = np.max(window) / (np.mean(window) + 1e-8)
    # spike vs sustained effort; bad form (momentum) = high spike, low mean = high ratio

    smoothness = np.mean(np.abs(np.diff(window, n=2))) if len(window) > 2 else 0.0
    # 2nd derivative = choppiness; distinct from wl (total path length)

    # rep shape: activation slopes
    # rise: avg slope from rep start to peak (bad form rises sharply via momentum)
    rise_slope = (window[peak_idx] - window[0]) / (peak_idx + 1e-8)
    # fall: avg slope from peak back to rep end (controlled descent = good form)
    fall_slope = (window[peak_idx] - window[-1]) / ((len(window) - peak_idx) + 1e-8)

    # number of secondary peaks — bad form (bounce/momentum) produces multiple humps
    if len(window) > 2:
        diff1 = np.diff(window)
        sign_changes = np.diff(np.sign(diff1))
        n_peaks = int(np.sum(sign_changes < 0))  # count downward sign changes = local maxima
    else:
        n_peaks = 0

    # Hjorth complexity — how much the frequency content changes over time;
    # choppy bad-form activation is more complex than smooth good-form
    if len(window) > 2:
        d1 = np.diff(window)
        d2 = np.diff(window, n=2)
        mob_signal = np.sqrt(np.var(d1) / (np.var(window) + 1e-8))
        mob_d1     = np.sqrt(np.var(d2) / (np.var(d1)     + 1e-8))
        complexity = mob_d1 / (mob_signal + 1e-8)
    else:
        complexity = 0.0

    return [mav, wl, var, skew, mean_freq,
            peak_loc, peak_ratio, smoothness,
            rise_slope, fall_slope,
            n_peaks, complexity]

def build_feature_matrix(windows, labels, sessions):
    features = [extract_features(w) for w in windows]
    X = np.array(features)
    y = np.array(labels)

    return X, y, sessions

def segment_reps(df, min_rep_samples=150, min_rest_samples=100, win=50):
    rep_signals = []
    labels = []
    sessions = []

    for session, group in df.groupby('session'):
        signal = group['emg_filtered'].values
        label = group['label'].iloc[0]

        envelope = group['emg_envelope'].values
        rms = [np.sqrt(np.mean(signal[i:i+win]**2)) for i in range(0, len(signal)-win, win)]
        threshold = np.percentile(rms, 60)
        above = [r > threshold for r in rms]

        in_rep = False
        rep_start = None
        last_rep_end = 0

        for i, a in enumerate(above):
            sample = i * win
            if not in_rep and a and (sample - last_rep_end) >= min_rest_samples:
                in_rep = True
                rep_start = sample
            elif in_rep and not a:
                rep_len = sample - rep_start
                if rep_len >= min_rep_samples:
                    rep_signals.append(envelope[rep_start:sample])
                    labels.append(label)
                    sessions.append(session)
                    last_rep_end = sample
                in_rep = False

        if in_rep and (len(signal) - rep_start) >= min_rep_samples:
            rep_signals.append(envelope[rep_start:])
            labels.append(label)
            sessions.append(session)

    return rep_signals, np.array(labels), np.array(sessions)

def normalize_features(X, sessions):
    X_norm = X.copy()
    for session in np.unique(sessions):
        mask = sessions == session
        X_norm[mask] = (X[mask] - X[mask].mean(axis=0)) / (X[mask].std(axis=0) + 1e-8)
    return X_norm

if __name__ == '__main__':
    df = load_sessions('../emg_sessions')
    df = preprocess(df)
    windows, labels, sessions = create_windows(df)
    X, y, sessions = build_feature_matrix(windows, labels, sessions)


    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Labels: {np.unique(y)}')



