import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import load_sessions, preprocess, segment_reps, extract_features

# These sessions are held out as a true test set.
# Do NOT tune features, thresholds, or hyperparameters based on these results.
# Evaluate on them only once, when you're done making decisions.
TEST_SESSIONS = {
    'good_20260410_172741',
    'bad_20260410_172942',
}

def cross_validate(X, y, sessions):
    unique_sessions = np.unique(sessions)
    results = []

    for test_session in unique_sessions:
        test_mask = sessions == test_session
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = clf.score(X_test, y_test)
        results.append((test_session, acc))
        print(f'{test_session}: {acc:.2f}')
        print(classification_report(y_test, y_pred, zero_division=0))

    avg = np.mean([r[1] for r in results])
    print(f'\nAverage CV accuracy: {avg:.2f}')
    return avg

def evaluate_test_set(X_dev, y_dev, X_test, y_test):
    mean = X_dev.mean(axis=0)
    std = X_dev.std(axis=0) + 1e-8
    X_dev_norm = (X_dev - mean) / std
    X_test_norm = (X_test - mean) / std

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_dev_norm, y_dev)
    y_pred = clf.predict(X_test_norm)

    print('\n--- FINAL TEST SET RESULTS ---')
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == '__main__':
    df = load_sessions('../emg_sessions')
    df = preprocess(df)
    rep_signals, y, sessions = segment_reps(df)

    X = np.array([extract_features(r) for r in rep_signals])
    print(f'Total reps: {len(rep_signals)}')
    print(f'Labels: {np.unique(y, return_counts=True)}')

    # Split into dev (CV) and held-out test
    test_mask = np.isin(sessions, list(TEST_SESSIONS))
    dev_mask = ~test_mask

    X_dev, y_dev, sessions_dev = X[dev_mask], y[dev_mask], sessions[dev_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f'\nDev reps: {len(y_dev)} | Test reps: {len(y_test)}')
    print(f'Test sessions: {TEST_SESSIONS}\n')

    # Feature importance on dev set only
    clf_full = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    mean = X_dev.mean(axis=0)
    std = X_dev.std(axis=0) + 1e-8
    clf_full.fit((X_dev - mean) / std, y_dev)
    feat_names = ['mav','wl','var','skew','mean_freq',
                  'peak_loc','peak_ratio','smoothness',
                  'rise_slope','fall_slope',
                  'n_peaks','complexity']
    importances = pd.Series(clf_full.feature_importances_, index=feat_names).sort_values(ascending=False)
    print(importances)

    # Cross-validate on dev set
    cross_validate(X_dev, y_dev, sessions_dev)

    # Uncomment the line below ONLY when done making decisions:
    # evaluate_test_set(X_dev, y_dev, X_test, y_test)

    # Save model and normalization stats for real-time inference
    joblib.dump({'model': clf_full, 'mean': mean, 'std': std}, 'model.joblib')
    print('\nModel saved to ml/model.joblib')
