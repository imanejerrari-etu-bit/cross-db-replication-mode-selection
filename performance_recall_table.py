"""
performance_recall_table.py -- Table 7 and Fig. 3 (Section 4.2):
PERFORMANCE-class recall versus the number of PERFORMANCE examples.

For each condition it reports
  Cases       total PERFORMANCE examples in the dataset
              (size-matched: mean over the 20 subsamples; 22 in 19, 21 in one)
  Train/fold  mean PERFORMANCE examples per training fold. In stratified
              5-fold CV every case is in exactly one test fold per repeat, so
              this is exactly Cases x 4/5.
  Recall      pooled over all repeated-CV test folds (total true positives /
              total PERFORMANCE test cases), the protocol of Table 5; the
              full-size values come from per_class_metrics.py and the
              size-matched value from sample_size_control.py.

Usage (after running per_class_metrics.py and sample_size_control.py):
    python performance_recall_table.py [--results-dir results]
"""

import argparse
import json
import os

import numpy as np

from common import LABEL_COLUMN, load_dataset

N_SPLITS = 5


def _load(results_dir: str, name: str) -> dict:
    for d in (results_dir, "."):
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise SystemExit(f"Missing {name}: run per_class_metrics.py and sample_size_control.py first.")


def main(results_dir: str) -> None:
    cases = {e: int((load_dataset(e)[LABEL_COLUMN] == "PERFORMANCE").sum())
             for e in ("postgresql", "mysql", "mongodb")}
    recall = {e: _load(results_dir, f"{e}_per_class_metrics.json")["classes"]["PERFORMANCE"]["recall"]
              for e in cases}
    ssc = _load(results_dir, "sample_size_control_summary.json")
    matched_cases = float(np.mean(ssc["performance_cases_per_subsample"]))
    frac = (N_SPLITS - 1) / N_SPLITS

    table = [
        ("PostgreSQL, full (n=644)", cases["postgresql"], recall["postgresql"]),
        ("PostgreSQL, matched (n=200)", matched_cases, ssc["pooled_performance_recall"]),
        ("MySQL (n=200)", cases["mysql"], recall["mysql"]),
        ("MongoDB (n=200)", cases["mongodb"], recall["mongodb"]),
    ]
    print(f"{'Condition':30s} {'Cases':>7s} {'Train/fold':>11s} {'Recall':>8s}")
    for name, n, r in table:
        print(f"{name:30s} {n:7.1f} {n * frac:11.1f} {r * 100:7.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    main(ap.parse_args().results_dir)
