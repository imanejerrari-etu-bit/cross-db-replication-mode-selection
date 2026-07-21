"""
train_classifiers.py — RQ1: per-engine RandomForest training under
repeated stratified 5-fold cross-validation.

Reproduces Table 3 (main-results) and, per engine, the 50 fold-level
accuracy estimates used later by stats_tests.py for the Mann-Whitney
comparisons (Section 3.3).

Usage:
    python src/train_classifiers.py --engine postgresql
    python src/train_classifiers.py --engine mysql
    python src/train_classifiers.py --engine mongodb
    python src/train_classifiers.py --engine all      # runs all three, saves results/*.npy
"""

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from common import load_dataset, make_classifier, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE

N_SPLITS = 5
N_REPEATS = 10  # -> 50 fold-level accuracy estimates per engine, matching the paper


def run_cv(engine: str, out_dir: str = "results") -> dict:
    df = load_dataset(engine)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    fold_accuracies = []
    fold_macro_f1 = []  # computed on CONSISTENCY/ECONOMIC/PERFORMANCE only, see note below
    pooled_true = []
    pooled_pred = []

    non_availability = ["CONSISTENCY", "ECONOMIC", "PERFORMANCE"]

    for train_idx, test_idx in rskf.split(X, y):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]

        fold_accuracies.append(accuracy_score(y_true, y_pred))

        # Macro-F1 excluding AVAILABILITY (Table 3, footnote 1): with
        # AVAILABILITY represented by 1-5 samples total, a handful of
        # folds will contain zero AVAILABILITY test examples, making
        # per-fold F1 for that class undefined. We therefore restrict
        # macro-F1 to the three non-scarce classes, exactly as stated
        # in the paper.
        mask = np.isin(y_true, non_availability)
        if mask.sum() > 0:
            fold_macro_f1.append(
                f1_score(y_true[mask], y_pred[mask], labels=non_availability,
                         average="macro", zero_division=0)
            )
            pooled_true.append(y_true[mask])
            pooled_pred.append(y_pred[mask])

    # Alternative macro-F1: computed ONCE on all pooled (concatenated)
    # fold predictions, rather than averaged per-fold. If your original
    # code reports macro-F1 this way instead, this number -- not
    # mean_macro_f1_3class above -- is the one to compare against the
    # paper. Per-fold averaging is noisier for small classes (e.g.
    # PERFORMANCE, ECONOMIC on MySQL/MongoDB) because each fold has
    # very few test examples of that class.
    pooled_macro_f1 = f1_score(
        np.concatenate(pooled_true), np.concatenate(pooled_pred),
        labels=non_availability, average="macro", zero_division=0,
    )
    # Weighted-F1 variant: weights each class's F1 by its support
    # (frequency) rather than treating all three classes equally. If
    # your original code used average="weighted" instead of "macro",
    # this number is the one to compare -- weighted F1 is pulled
    # toward the dominant class's (near-perfect) score, so it will be
    # higher than macro-F1 especially where the 3-class subset is
    # itself imbalanced (MySQL, MongoDB more than PostgreSQL).
    pooled_weighted_f1 = f1_score(
        np.concatenate(pooled_true), np.concatenate(pooled_pred),
        labels=non_availability, average="weighted", zero_division=0,
    )

    result = {
        "engine": engine,
        "n": len(df),
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "mean_macro_f1_3class_per_fold_avg": float(np.mean(fold_macro_f1)),
        "macro_f1_3class_pooled": float(pooled_macro_f1),
        "weighted_f1_3class_pooled": float(pooled_weighted_f1),
        "n_folds": len(fold_accuracies),
    }

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{engine}_fold_accuracies.npy"), np.array(fold_accuracies))
    with open(os.path.join(out_dir, f"{engine}_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["postgresql", "mysql", "mongodb", "all"],
                         required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    engines = ["postgresql", "mysql", "mongodb"] if args.engine == "all" else [args.engine]
    for eng in engines:
        res = run_cv(eng, args.out_dir)
        print(f"{eng:12s}  n={res['n']:4d}  "
              f"mean_accuracy={res['mean_accuracy']*100:.2f}%  "
              f"macro_f1_per-fold-avg={res['mean_macro_f1_3class_per_fold_avg']*100:.2f}%  "
              f"macro_f1_pooled={res['macro_f1_3class_pooled']*100:.2f}%  "
              f"weighted_f1_pooled={res['weighted_f1_3class_pooled']*100:.2f}%  "
              f"folds={res['n_folds']}")
