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
    # time 
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    wl = np.sum(np.abs(np.diff(window)))
    var = np.var(window)
    skew = pd.Series(window).skew()
    kurt = pd.Series(window).kurt()

    # freq
    freqs = np.fft.rfftfreq(len(window), d=1/fs)
    power = np.abs(np.fft.rfft(window))**2 
    mean_freq = np.sum(freqs*power) / np.sum(power)
    median_freq = freqs[np.searchsorted(np.cumsum(power), np.sum(power) / 2)]
    return [rms, mav, zc, wl, var, skew, kurt, mean_freq, median_freq]

def build_feature_matrix(windows, labels, sessions):
    features = [extract_features(w) for w in windows]
    X = np.array(features)
    y = np.array(labels)

    return X, y, sessions

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



