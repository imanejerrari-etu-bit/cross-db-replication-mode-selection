"""
per_class_metrics.py — C13 (Reviewer 2): per-class precision/recall for
CONSISTENCY and ECONOMIC (and, for completeness, PERFORMANCE and
AVAILABILITY), computed on the SAME pooled predictions used for the
pooled macro-F1 in Table 3 (train_classifiers.py), so the numbers here
are directly consistent with that table rather than a separate run.

Usage:
    python per_class_metrics.py --engine all --out-dir results
"""

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

from common import load_dataset, make_classifier, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE

N_SPLITS = 5
N_REPEATS = 10
ALL_CLASSES = ["CONSISTENCY", "ECONOMIC", "PERFORMANCE", "AVAILABILITY"]


def run_per_class(engine: str, out_dir: str = "results") -> dict:
    df = load_dataset(engine)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    pooled_true, pooled_pred = [], []
    for train_idx, test_idx in rskf.split(X, y):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        pooled_pred.append(clf.predict(X[test_idx]))
        pooled_true.append(y[test_idx])

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_CLASSES, average=None, zero_division=0
    )

    result = {
        "engine": engine,
        "classes": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(ALL_CLASSES)
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{engine}_per_class_metrics.json"), "w") as f:
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
        res = run_per_class(eng, args.out_dir)
        print(f"\n=== {eng} ===")
        print(f"{'Class':14s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
        for cls, m in res["classes"].items():
            print(f"{cls:14s} {m['precision']*100:9.1f}% {m['recall']*100:7.1f}% "
                  f"{m['f1']*100:7.1f}% {m['support']:8d}")
