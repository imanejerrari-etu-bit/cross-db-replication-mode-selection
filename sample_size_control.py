"""
sample_size_control.py — RQ1b (Section 3.4, Section 4.2): disentangles
engine identity from training-set size by evaluating 20 independent
class-proportional stratified subsamples of PostgreSQL at n=200,
under the SAME repeated-CV protocol as the full dataset (10 repeats x
5 folds = 50 fold-level estimates per subsample), exactly matching
Section 3.2's Classification Protocol.

This produces 20 subsamples x 50 estimates = 1000 fold-level accuracy
estimates in total, aggregated for Table 5 / Section 4.2, and the
PERFORMANCE-recall breakdown in Table 6.

NOTE ON A PREVIOUS DRAFT INCONSISTENCY: an earlier version of the
paper text claimed this protocol yields "40 fold-level accuracy
estimates" while also saying it uses "the same repeated stratified
cross-validation protocol as the full dataset" (10x5=50 per
subsample) -- these two claims are mutually inconsistent (20 x 50 =
1000, not 40). This script resolves the ambiguity by literally
implementing the second claim (same protocol as the full dataset).
If you locate the original code and it used a different protocol,
update N_REPEATS_PER_SUBSAMPLE below AND the paper text to match
whatever that original protocol actually was.

Usage:
    python src/sample_size_control.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, recall_score

from common import load_dataset, make_classifier, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE

N_SUBSAMPLES = 20
MATCHED_N = 200
# 10 repeats x 5 folds per subsample x 20 subsamples = 1000 fold
# estimates total -- this is deliberately identical to the protocol
# used for the full-dataset results in train_classifiers.py.
N_REPEATS_PER_SUBSAMPLE = 10
N_SPLITS = 5


def stratified_subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Class-proportional stratified subsample of size n, matching the
    class distribution documented in Table 2 (158 CONSISTENCY,
    22 PERFORMANCE, 18 ECONOMIC, 2 AVAILABILITY for n=200)."""
    frac = n / len(df)
    parts = []
    for cls, group in df.groupby(LABEL_COLUMN):
        k = max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(k, len(group)), random_state=rng.integers(1e9)))
    out = pd.concat(parts)
    # Trim/pad to exactly n if rounding drifted, preserving class ratios as closely as possible
    if len(out) > n:
        out = out.sample(n=n, random_state=rng.integers(1e9))
    return out.reset_index(drop=True)


def evaluate_subsample(df: pd.DataFrame, rng_seed: int) -> dict:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS_PER_SUBSAMPLE, random_state=rng_seed
    )

    accuracies = []
    perf_recalls = []
    for train_idx, test_idx in rskf.split(X, y):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_true = y[test_idx]
        accuracies.append(accuracy_score(y_true, y_pred))
        if (y_true == "PERFORMANCE").sum() > 0:
            perf_recalls.append(
                recall_score(y_true, y_pred, labels=["PERFORMANCE"], average="micro")
            )

    return {"accuracies": accuracies, "performance_recalls": perf_recalls}


def run(out_dir: str = "results") -> dict:
    full_df = load_dataset("postgresql")
    rng = np.random.default_rng(RANDOM_STATE)

    all_accuracies = []
    all_perf_recalls = []

    for i in range(N_SUBSAMPLES):
        sub = stratified_subsample(full_df, MATCHED_N, rng)
        res = evaluate_subsample(sub, rng_seed=RANDOM_STATE + i)
        all_accuracies.extend(res["accuracies"])
        all_perf_recalls.extend(res["performance_recalls"])

    result = {
        "condition": "postgresql_size_matched_n200",
        "n_subsamples": N_SUBSAMPLES,
        "n_fold_estimates": len(all_accuracies),
        "mean_accuracy": float(np.mean(all_accuracies)),
        "mean_performance_recall": float(np.mean(all_perf_recalls)) if all_perf_recalls else None,
    }

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "postgresql_matched_fold_accuracies.npy"),
            np.array(all_accuracies))
    with open(os.path.join(out_dir, "sample_size_control_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = run()
    print(f"PostgreSQL, size-matched (n={MATCHED_N}): "
          f"{res['n_fold_estimates']} fold estimates from "
          f"{res['n_subsamples']} subsamples")
    print(f"  mean accuracy           = {res['mean_accuracy']*100:.2f}%")
    if res["mean_performance_recall"] is not None:
        print(f"  mean PERFORMANCE recall = {res['mean_performance_recall']*100:.1f}%")
    print("\nCompare against results/{postgresql,mysql,mongodb}_fold_accuracies.npy "
          "(from train_classifiers.py) using stats_tests.mann_whitney_between().")
