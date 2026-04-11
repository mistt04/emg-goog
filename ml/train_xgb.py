import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report
from preprocess import load_sessions, preprocess, segment_reps, extract_features

# Same held-out test sessions as train.py — do NOT tune based on these.
TEST_SESSIONS = {
    'good_20260410_172741',
    'bad_20260410_172942',
}

FEAT_NAMES = ['mav', 'wl', 'var', 'skew', 'mean_freq',
              'peak_loc', 'peak_ratio', 'smoothness',
              'rise_slope', 'fall_slope',
              'n_peaks', 'complexity']

XGB_DEVICE = {}  # GPU slower than CPU on this dataset size; set {'device':'cuda','tree_method':'hist'} to try


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def _evaluate_combo(params, X, y_int, session_groups):
    """Run LOGO CV for one param combo. Called in parallel."""
    logo = LeaveOneGroupOut()
    fold_scores = []
    for train_idx, test_idx in logo.split(X, y_int, groups=session_groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = XGBClassifier(
            **params,
            n_jobs=1,             # 1 thread per fit — cores used at the combo level
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        fold_scores.append(np.mean(clf.predict(X_te) == y_te))
    return np.mean(fold_scores)


def tune_xgboost(X, y_int, session_groups):
    """Random search over XGBoost params using leave-one-session-out CV."""
    param_grid = {
        'max_depth':        [3, 4, 5],
        'learning_rate':    [0.02, 0.05, 0.1],
        'n_estimators':     [200, 400],
        'subsample':        [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
    }

    from itertools import product
    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, values)) for values in product(*param_grid.values())]

    print(f'Grid search: {len(combos)} combos across {len(param_grid)} params')
    scores = Parallel(n_jobs=-1)(
        delayed(_evaluate_combo)(params, X, y_int, session_groups)
        for params in tqdm(combos, desc='Tuning XGBoost', unit='combo')
    )

    best_idx    = int(np.argmax(scores))
    best_params = combos[best_idx]
    best_score  = scores[best_idx]

    print(f'\nBest params: {best_params}')
    print(f'Best CV accuracy: {best_score:.3f}\n')
    return best_params


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _label_encode(y):
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    return y_int, le


def cross_validate_xgb(X, y, sessions, xgb_params):
    """Leave-one-session-out CV for XGBoost with best params."""
    print('--- XGBoost CV (best params) ---')
    y_int, le = _label_encode(y)
    unique_sessions = np.unique(sessions)
    results = []

    for test_session in unique_sessions:
        test_mask  = sessions == test_session
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_int[train_mask], y_int[test_mask]

        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)

        clf = XGBClassifier(
            **xgb_params,
            scale_pos_weight=n_neg / (n_pos + 1e-8),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            **XGB_DEVICE,
        )
        clf.fit(X_train, y_train)
        train_acc = np.mean(clf.predict(X_train) == y_train)
        val_acc   = np.mean(clf.predict(X_test)  == y_test)
        results.append((test_session, train_acc, val_acc))
        print(f'{test_session}:  train={train_acc:.2f}  val={val_acc:.2f}  gap={train_acc - val_acc:+.2f}')

    avg_train = np.mean([r[1] for r in results])
    avg_val   = np.mean([r[2] for r in results])
    print(f'\nAverage train accuracy: {avg_train:.3f}')
    print(f'Average val accuracy:   {avg_val:.3f}')
    print(f'Average gap:            {avg_train - avg_val:+.3f}\n')
    return avg_val


def evaluate_test_set(X_dev, y_dev, X_test, y_test, xgb_params):
    y_dev_int, le = _label_encode(y_dev)

    mean = X_dev.mean(axis=0)
    std  = X_dev.std(axis=0) + 1e-8
    X_dev_norm  = (X_dev  - mean) / std
    X_test_norm = (X_test - mean) / std

    n_neg = np.sum(y_dev_int == 0)
    n_pos = np.sum(y_dev_int == 1)

    clf = XGBClassifier(
        **xgb_params,
        scale_pos_weight=n_neg / (n_pos + 1e-8),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_dev_norm, y_dev_int)
    y_pred = le.inverse_transform(clf.predict(X_test_norm))

    print('\n--- FINAL TEST SET RESULTS ---')
    print(classification_report(y_test, y_pred, zero_division=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df = load_sessions('../emg_sessions')
    df = preprocess(df)
    rep_signals, y, sessions = segment_reps(df)

    X = np.array([extract_features(r) for r in rep_signals])
    print(f'Total reps: {len(rep_signals)}')
    print(f'Features per rep: {X.shape[1]}  ({", ".join(FEAT_NAMES)})')
    print(f'Labels: {np.unique(y, return_counts=True)}\n')

    # Split dev / held-out test
    test_mask = np.isin(sessions, list(TEST_SESSIONS))
    dev_mask  = ~test_mask

    X_dev, y_dev, sessions_dev = X[dev_mask], y[dev_mask], sessions[dev_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f'Dev reps: {len(y_dev)} | Test reps: {len(y_test)}')
    print(f'Test sessions: {TEST_SESSIONS}\n')

    y_dev_int, le   = _label_encode(y_dev)
    session_groups  = LabelEncoder().fit_transform(sessions_dev)

    # 1. Tune
    xgb_params = tune_xgboost(X_dev, y_dev_int, session_groups)

    # 2. Feature importance on full dev set with best params
    mean = X_dev.mean(axis=0)
    std  = X_dev.std(axis=0) + 1e-8
    clf_full = XGBClassifier(
        **xgb_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    clf_full.fit((X_dev - mean) / std, y_dev_int)
    importances = pd.Series(clf_full.feature_importances_, index=FEAT_NAMES).sort_values(ascending=False)
    print('Feature importances (dev set):')
    print(importances, '\n')

    # 3. CV
    cross_validate_xgb(X_dev, y_dev, sessions_dev, xgb_params)

    # 4. Final test set evaluation
    evaluate_test_set(X_dev, y_dev, X_test, y_test, xgb_params)

    # 5. Save model and normalization stats for real-time inference
    joblib.dump({
        'model': clf_full,
        'mean': mean,
        'std': std,
        'label_map': {i: lbl for i, lbl in enumerate(le.classes_)},
        'feat_names': FEAT_NAMES,
        'xgb_params': xgb_params,
    }, 'model_xgb.joblib')
    print('Model saved to ml/model_xgb.joblib')
