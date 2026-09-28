"""
compare_classifiers.py — RQ1c: is RandomForest competitive with
alternative classifiers on this specific four-class mode-selection
task? (Section 3.4, Table 7/8 of the paper.)

Trains RandomForest, XGBoost, multinomial LogisticRegression, and an
RBF-kernel SVM under the IDENTICAL protocol used in train_classifiers.py
(same features, same RepeatedStratifiedKFold with random_state=42,
same 10 repeats x 5 folds -> 50 fold-level estimates per engine per
classifier), so that Table 7 (accuracy) and Table 8 (pooled 3-class
macro-F1) are directly comparable to Table 3.

XGBoost configuration: n_estimators=200 (to match RandomForest's
200 trees, the only hyperparameter explicitly fixed to match across
models per Section 3.4), with all other hyperparameters left at the
installed xgboost library's defaults (see XGBOOST_VERSION below) --
no additional tuning was performed for either model, consistent with
this comparison's goal of testing out-of-the-box competitiveness
rather than a tuned comparison.

LogisticRegression and SVM(RBF): features standardised (zero mean,
unit variance) via StandardScaler fit on each training fold only
(never on test folds, to avoid leakage), both with
class_weight="balanced" to match RandomForest's balanced weighting
under the same class-imbalance conditions (Section 3.2).

Usage:
    python compare_classifiers.py --engine postgresql
    python compare_classifiers.py --engine all --out-dir results
"""

import argparse
import json
import os

import numpy as np
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import load_dataset, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE

N_SPLITS = 5
N_REPEATS = 10  # -> 50 fold-level estimates per engine, matching train_classifiers.py
NON_AVAILABILITY = ["CONSISTENCY", "ECONOMIC", "PERFORMANCE"]
XGBOOST_VERSION = xgboost.__version__


def make_models():
    """One fresh, unfitted estimator per classifier per fold. Returns
    a dict name -> constructor (called once per fold to avoid any
    state leaking across folds)."""
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": lambda: _xgb_pipeline(),
        "LogisticRegression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000, random_state=RANDOM_STATE,
            )),
        ]),
        "SVM (RBF)": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def _xgb_pipeline():
    # XGBoost needs integer-encoded labels; wrap with a small label
    # encoder step so the public interface still takes/returns the
    # original string class names, exactly like the other three models.
    from sklearn.preprocessing import LabelEncoder

    class _XGBStringLabels:
        def __init__(self):
            self.le = LabelEncoder()
            self.model = XGBClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )

        def fit(self, X, y):
            y_enc = self.le.fit_transform(y)
            self.model.fit(X, y_enc)
            return self

        def predict(self, X):
            pred_enc = self.model.predict(X)
            return self.le.inverse_transform(pred_enc)

    return _XGBStringLabels()


def run_comparison(engine: str, out_dir: str = "results") -> dict:
    df = load_dataset(engine)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    # Materialise the folds once so every classifier sees the IDENTICAL
    # train/test splits (required for a fair, paired comparison and for
    # any future paired significance test between classifiers).
    folds = list(rskf.split(X, y))

    results = {}
    for name, ctor in make_models().items():
        fold_acc = []
        pooled_true, pooled_pred = [], []
        for train_idx, test_idx in folds:
            clf = ctor()
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            y_true = y[test_idx]
            fold_acc.append(accuracy_score(y_true, y_pred))
            mask = np.isin(y_true, NON_AVAILABILITY)
            if mask.sum() > 0:
                pooled_true.append(y_true[mask])
                pooled_pred.append(y_pred[mask])
        macro_f1 = f1_score(
            np.concatenate(pooled_true), np.concatenate(pooled_pred),
            labels=NON_AVAILABILITY, average="macro", zero_division=0,
        )
        results[name] = {
            "mean_accuracy": float(np.mean(fold_acc)),
            "std_accuracy": float(np.std(fold_acc)),
            "macro_f1_pooled_3class": float(macro_f1),
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{engine}_classifier_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"engine": engine, "xgboost_version": XGBOOST_VERSION,
                    "results": results}, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["postgresql", "mysql", "mongodb", "all"],
                         required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    engines = ["postgresql", "mysql", "mongodb"] if args.engine == "all" else [args.engine]
    for eng in engines:
        res = run_comparison(eng, args.out_dir)
        print(f"\n=== {eng} (xgboost {XGBOOST_VERSION}) ===")
        for clf_name, r in res.items():
            print(f"  {clf_name:20s}  acc={r['mean_accuracy']*100:6.2f}%  "
                  f"macro_f1={r['macro_f1_pooled_3class']*100:6.2f}%")
