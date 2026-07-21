"""
smote_availability.py — RQ3 (Section 4.5): SMOTE oversampling
diagnostic on PostgreSQL's AVAILABILITY class, the only engine with
enough AVAILABILITY samples (n=5) to make oversampling meaningful.

IMPORTANT — k_neighbors handling:
SMOTE's default k_neighbors=5 requires at least 6 minority-class
samples to interpolate from. PostgreSQL has only 5 AVAILABILITY
samples in the FULL dataset, and each 5-fold CV training split holds
out ~1/5 of them, leaving as few as 3-4 in training. This script
therefore sets k_neighbors adaptively per fold:
    k_neighbors = max(1, min(5, n_minority_in_training_fold - 1))
and SKIPS SMOTE entirely on any fold where the training data has
fewer than 2 AVAILABILITY samples (SMOTE cannot interpolate from a
single point). This has to be stated explicitly in the paper's
Methodology (Section 3.3) or RQ3 (Section 4.5) — whichever
k_neighbors value your actual reported 20% recall / F1=0.29 came
from, the text must say so numerically. Run this script, note the
printed k_neighbors value(s), and put that number in the paper.

Usage:
    python src/smote_availability.py
"""

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

from common import load_dataset, make_classifier, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE


def run(out_dir: str = "results") -> dict:
    df = load_dataset("postgresql")
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_true, all_pred = [], []
    k_neighbors_used = []
    skipped_folds = 0

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_minority = int((y_train == "AVAILABILITY").sum())

        if n_minority < 2:
            # Cannot interpolate from fewer than 2 points; train on
            # the original (imbalanced) fold instead of failing.
            skipped_folds += 1
            clf = make_classifier()
            clf.fit(X_train, y_train)
        else:
            k = max(1, min(5, n_minority - 1))
            k_neighbors_used.append(k)
            sm = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            clf = make_classifier()
            clf.fit(X_res, y_res)

        all_true.append(y_test)
        all_pred.append(clf.predict(X_test))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    result = {
        "engine": "postgresql",
        "k_neighbors_per_fold": k_neighbors_used,
        "folds_skipped_too_few_minority": skipped_folds,
        "availability_recall": float(
            recall_score(y_true, y_pred, labels=["AVAILABILITY"], average="micro", zero_division=0)
        ),
        "availability_precision": float(
            precision_score(y_true, y_pred, labels=["AVAILABILITY"], average="micro", zero_division=0)
        ),
        "availability_f1": float(
            f1_score(y_true, y_pred, labels=["AVAILABILITY"], average="micro", zero_division=0)
        ),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smote_availability_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = run()
    print(f"k_neighbors used per fold (skipping folds with <2 minority samples): "
          f"{res['k_neighbors_per_fold']}")
    print(f"folds skipped (too few AVAILABILITY in training): "
          f"{res['folds_skipped_too_few_minority']}")
    print(f"AVAILABILITY recall    = {res['availability_recall']*100:.1f}%")
    print(f"AVAILABILITY precision = {res['availability_precision']*100:.1f}%")
    print(f"AVAILABILITY F1        = {res['availability_f1']:.2f}")
    print("\nIMPORTANT: update Section 4.5 of the paper with the actual "
          "k_neighbors value(s) printed above, and re-verify the recall/"
          "precision/F1 figures against what is currently written "
          "(20% recall, F1=0.29, 50% precision) -- they may shift "
          "slightly depending on the exact k_neighbors and fold split used.")
