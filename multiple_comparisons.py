"""
multiple_comparisons.py -- Section 3.5 ("Multiple comparisons"), Tables 13
and the adjusted p-values reported in Sections 4.1, 4.2 and 4.7.

The Mann-Whitney comparisons of the paper are grouped into three families
of three tests each:

  F1  pairwise engine-accuracy comparisons (50 fold-level accuracies per
      engine, Section 4.1)
  F2  size-matched PostgreSQL (1000 fold-level accuracies) against
      full-size PostgreSQL, MySQL and MongoDB (Section 4.2)
  F3  pairwise total-RTO comparisons from the live failover tests
      (total RTO = promotion_time_s + rto_s, Section 4.7)

For every test the script reports the unadjusted two-sided p-value, the
Bonferroni-adjusted value min(1, 3p) and the Holm-adjusted value within
the family. For F3 it also reports the rank-biserial correlation
r = 1 - 2U / (n1 * n2) for the first-listed group versus the second.
As a sensitivity check it also reports Bonferroni adjustment over all
nine tests pooled into one family.

Inputs (produced by the other scripts, run from the repository root):
  results/{postgresql,mysql,mongodb}_fold_accuracies.npy   train_classifiers.py
  results/postgresql_matched_fold_accuracies.npy           sample_size_control.py
  rto_postgres_30tests.csv, rto_mysql_clean.csv, rto_mongodb_clean.csv

Usage:
    python train_classifiers.py --engine all --out-dir results
    python sample_size_control.py
    python multiple_comparisons.py

Adjusted values are computed from unrounded p-values and may differ in the
last digit from values recomputed from the rounded figures in the text.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS_DIR = "results"
RTO_FILES = {
    "PostgreSQL": "rto_postgres_30tests.csv",
    "MySQL": "rto_mysql_clean.csv",
    "MongoDB": "rto_mongodb_clean.csv",
}


def _load_npy(name: str) -> np.ndarray:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        sys.exit(
            f"Missing {path}. Run train_classifiers.py (all engines) and "
            "sample_size_control.py first."
        )
    return np.load(path)


def _total_rto(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return (df["promotion_time_s"] + df["rto_s"]).to_numpy()


def mw(a: np.ndarray, b: np.ndarray) -> dict:
    res = mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    return {
        "U": u,
        "p": float(res.pvalue),
        "rank_biserial": 1.0 - 2.0 * u / (len(a) * len(b)),
        "n1": len(a),
        "n2": len(b),
    }


def adjust(ps: list) -> tuple:
    """Bonferroni min(1, m*p) and Holm step-down adjusted p-values."""
    m = len(ps)
    bonf = [min(1.0, m * p) for p in ps]
    holm = [0.0] * m
    running = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        holm[i] = running
    return bonf, holm


def run_family(name: str, comparisons: list) -> list:
    rows = [dict(label=lab, **mw(a, b)) for lab, a, b in comparisons]
    bonf, holm = adjust([r["p"] for r in rows])
    for r, pb, ph in zip(rows, bonf, holm):
        r["p_bonferroni"], r["p_holm"] = pb, ph
        r["family"] = name
    print(f"\n{name}")
    print(f"{'comparison':32s} {'U':>9s} {'p':>10s} {'p_Bonf':>10s} {'p_Holm':>10s} {'r':>7s}")
    for r in rows:
        print(f"{r['label']:32s} {r['U']:9.1f} {r['p']:10.4g} "
              f"{r['p_bonferroni']:10.4g} {r['p_holm']:10.4g} {r['rank_biserial']:7.2f}")
    return rows


def main() -> None:
    acc = {e: _load_npy(f"{e}_fold_accuracies.npy") for e in ("postgresql", "mysql", "mongodb")}
    matched = _load_npy("postgresql_matched_fold_accuracies.npy")
    rto = {k: _total_rto(v) for k, v in RTO_FILES.items()}

    f1 = run_family("F1: engine accuracy (Section 4.1)", [
        ("MongoDB vs MySQL", acc["mongodb"], acc["mysql"]),
        ("MongoDB vs PostgreSQL", acc["mongodb"], acc["postgresql"]),
        ("MySQL vs PostgreSQL", acc["mysql"], acc["postgresql"]),
    ])
    f2 = run_family("F2: size-matched PostgreSQL (Section 4.2)", [
        ("matched vs PostgreSQL full", matched, acc["postgresql"]),
        ("matched vs MySQL native", matched, acc["mysql"]),
        ("matched vs MongoDB native", matched, acc["mongodb"]),
    ])
    f3 = run_family("F3: total RTO (Section 4.7, Table 13)", [
        ("PostgreSQL vs MySQL", rto["PostgreSQL"], rto["MySQL"]),
        ("PostgreSQL vs MongoDB", rto["PostgreSQL"], rto["MongoDB"]),
        ("MySQL vs MongoDB", rto["MySQL"], rto["MongoDB"]),
    ])

    print("\nSensitivity check: Bonferroni over all nine tests pooled (factor 9)")
    for r in f1 + f2 + f3:
        p9 = min(1.0, 9 * r["p"])
        r["p_bonferroni_pooled9"] = p9
        print(f"  {r['family'][:2]} {r['label']:30s} p x 9 = {p9:9.4g}  "
              f"{'significant' if p9 < 0.05 else 'not significant'}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "multiple_comparisons.json"), "w") as f:
        json.dump(f1 + f2 + f3, f, indent=2)


if __name__ == "__main__":
    main()
