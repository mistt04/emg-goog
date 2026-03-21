import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_sessions, preprocess, create_windows, build_feature_matrix, normalize_features

def cross_validate(X, y, sessions):
    unique_sessions = np.unique(sessions)
    results = []

    for test_session in unique_sessions:
        test_mask = sessions == test_session
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

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
    windows, labels, sessions = create_windows(df)
    X, y, sessions = build_feature_matrix(windows, labels, sessions)
    X = normalize_features(X, sessions)
    cross_validate(X, y, sessions)


    


