import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from preprocess import load_sessions, preprocess, segment_reps, extract_features

# Must match train_xgb.py — do not include test sessions
TEST_SESSIONS = {
    'good_20260410_172741',
    'bad_20260410_172942',
}

XGB_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
}

N_PERMUTATIONS = 200


def logo_cv_accuracy(X, y_int, session_groups):
    """Run LOGO CV and return mean accuracy."""
    logo = LeaveOneGroupOut()
    scores = []
    for train_idx, test_idx in logo.split(X, y_int, groups=session_groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        n_neg = np.sum(y_tr == 0)
        n_pos = np.sum(y_tr == 1)

        clf = XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=n_neg / (n_pos + 1e-8),
            n_jobs=1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        scores.append(np.mean(clf.predict(X_te) == y_te))
    return np.mean(scores)


def _permutation_run(X, y_int, session_groups, seed):
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_int)
    return logo_cv_accuracy(X, y_shuffled, session_groups)


if __name__ == '__main__':
    df = load_sessions('../emg_sessions')
    df = preprocess(df)
    rep_signals, y, sessions = segment_reps(df)

    X = np.array([extract_features(r) for r in rep_signals])

    # Exclude held-out test sessions
    dev_mask = ~np.isin(sessions, list(TEST_SESSIONS))
    X, y, sessions = X[dev_mask], y[dev_mask], sessions[dev_mask]

    le = LabelEncoder()
    y_int = le.fit_transform(y)
    session_groups = LabelEncoder().fit_transform(sessions)

    # Real accuracy
    print('Running real LOGO CV...')
    real_acc = logo_cv_accuracy(X, y_int, session_groups)
    print(f'Real accuracy: {real_acc:.3f}\n')

    # Permutation distribution
    print(f'Running {N_PERMUTATIONS} permutations in parallel...')
    perm_accs = Parallel(n_jobs=-1)(
        delayed(_permutation_run)(X, y_int, session_groups, seed)
        for seed in tqdm(range(N_PERMUTATIONS), unit='perm')
    )
    perm_accs = np.array(perm_accs)

    p_value = np.mean(perm_accs >= real_acc)

    print(f'\n--- Permutation Test Results ---')
    print(f'Real accuracy:        {real_acc:.3f}')
    print(f'Permuted mean:        {perm_accs.mean():.3f}')
    print(f'Permuted std:         {perm_accs.std():.3f}')
    print(f'Permuted max:         {perm_accs.max():.3f}')
    print(f'p-value:              {p_value:.3f}')
    print(f'\n{"Model is learning real signal." if p_value < 0.05 else "Result may be chance — investigate further."}')
