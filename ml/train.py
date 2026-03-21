import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_sessions, preprocess, segment_reps, extract_features

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

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results.append((test_session, acc))
        print(f'{test_session}: {acc:.2f}')

    avg = np.mean([r[1] for r in results])
    print(f'\nAverage accuracy: {avg:.2f}')

if __name__ == '__main__':
    df = load_sessions('../emg_sessions')
    df = preprocess(df)
    rep_signals, y, sessions = segment_reps(df)

    X = np.array([extract_features(r) for r in rep_signals])
    print(f'Total reps: {len(rep_signals)}')
    print(f'Labels: {np.unique(y, return_counts=True)}')

    clf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_full.fit(X, y)
    feat_names = ['rms','mav','wl','var','skew','kurt','mean_freq']
    importances = pd.Series(clf_full.feature_importances_, index=feat_names).sort_values(ascending=False)
    print(importances)

    cross_validate(X, y, sessions)


    


