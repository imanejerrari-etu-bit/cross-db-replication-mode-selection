"""
stats_tests.py — Section 3.3 (Statistical Testing): BCa bootstrap 95%
confidence intervals, McNemar's test (RandomForest vs. static
heuristic, within engine), and Mann-Whitney U tests (between engines,
on fold-level accuracy distributions from train_classifiers.py).

Requires train_classifiers.py to have been run first (uses
results/{engine}_fold_accuracies.npy).

Usage:
    python src/train_classifiers.py --engine all
    python src/stats_tests.py
"""

import json
import os

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

from common import load_dataset, make_classifier, cpu_heuristic, \
    FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE

RESULTS_DIR = "results"
ENGINES = ["mongodb", "mysql", "postgresql"]


# ---------------------------------------------------------------
# BCa bootstrap 95% CI on mean accuracy (Table 3)
# ---------------------------------------------------------------
def bca_ci(fold_accuracies: np.ndarray, n_resamples: int = 10_000) -> tuple:
    res = stats.bootstrap(
        (fold_accuracies,), np.mean,
        n_resamples=n_resamples, method="BCa",
        random_state=RANDOM_STATE,
    )
    return res.confidence_interval.low, res.confidence_interval.high


# ---------------------------------------------------------------
# McNemar test: RandomForest vs. static heuristic, single stratified
# 5-fold split (paired predictions are required for McNemar's test;
# this is why Table 4's RF accuracy differs slightly from the
# repeated-CV mean in Table 3 — see the paper's note on this).
# ---------------------------------------------------------------
def mcnemar_vs_heuristic(engine: str) -> dict:
    df = load_dataset(engine)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_true, all_rf_pred, all_heur_pred = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        all_true.append(y[test_idx])
        all_rf_pred.append(clf.predict(X[test_idx]))
        all_heur_pred.append(cpu_heuristic(df.iloc[test_idx]))

    y_true = np.concatenate(all_true)
    rf_pred = np.concatenate(all_rf_pred)
    heur_pred = np.concatenate(all_heur_pred)

    rf_correct = (rf_pred == y_true)
    heur_correct = (heur_pred == y_true)

    # 2x2 contingency table for McNemar's test
    both_correct = np.sum(rf_correct & heur_correct)
    rf_only = np.sum(rf_correct & ~heur_correct)
    heur_only = np.sum(~rf_correct & heur_correct)
    both_wrong = np.sum(~rf_correct & ~heur_correct)
    table = [[both_correct, rf_only], [heur_only, both_wrong]]

    test_result = mcnemar(table, exact=(rf_only + heur_only < 25), correction=True)

    return {
        "engine": engine,
        "rf_accuracy": float(rf_correct.mean()),
        "heuristic_accuracy": float(heur_correct.mean()),
        "delta_pp": float((rf_correct.mean() - heur_correct.mean()) * 100),
        "mcnemar_statistic": float(test_result.statistic),
        "mcnemar_p": float(test_result.pvalue),
    }


# ---------------------------------------------------------------
# Mann-Whitney U on fold-level accuracy distributions (between engines)
# ---------------------------------------------------------------
def mann_whitney_between(engine_a: str, engine_b: str) -> dict:
    acc_a = np.load(os.path.join(RESULTS_DIR, f"{engine_a}_fold_accuracies.npy"))
    acc_b = np.load(os.path.join(RESULTS_DIR, f"{engine_b}_fold_accuracies.npy"))
    u_stat, p_val = stats.mannwhitneyu(acc_a, acc_b, alternative="two-sided")
    return {"pair": f"{engine_a} vs {engine_b}", "U": float(u_stat), "p": float(p_val)}


if __name__ == "__main__":
    print("=== BCa 95% bootstrap CIs (Table 3) ===")
    for eng in ENGINES:
        acc = np.load(os.path.join(RESULTS_DIR, f"{eng}_fold_accuracies.npy"))
        lo, hi = bca_ci(acc)
        print(f"{eng:12s}  mean={acc.mean()*100:.2f}%  "
              f"BCa 95% CI=[{lo*100:.2f}, {hi*100:.2f}]")

    print("\n=== McNemar: RandomForest vs. static heuristic (Table 4) ===")
    mcnemar_results = []
    for eng in ENGINES:
        r = mcnemar_vs_heuristic(eng)
        mcnemar_results.append(r)
        print(f"{eng:12s}  RF={r['rf_accuracy']*100:.1f}%  "
              f"Heur={r['heuristic_accuracy']*100:.1f}%  "
              f"Δ={r['delta_pp']:+.1f}pp  p={r['mcnemar_p']:.2e}")

    print("\n=== Mann-Whitney U: fold-level accuracy, between engines ===")
    pairs = [("mongodb", "mysql"), ("mongodb", "postgresql"), ("mysql", "postgresql")]
    mw_results = [mann_whitney_between(a, b) for a, b in pairs]
    for r in mw_results:
        print(f"{r['pair']:25s}  U={r['U']:.1f}  p={r['p']:.4g}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "stats_tests_summary.json"), "w") as f:
        json.dump({"mcnemar": mcnemar_results, "mann_whitney": mw_results}, f, indent=2)
