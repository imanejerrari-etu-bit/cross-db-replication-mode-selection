"""
analyze_rto_total.py — Section 4.4 (RQ4): live failover RTO analysis
across PostgreSQL, MySQL, and MongoDB.

IMPORTANT — why this script exists instead of the original
analyze_all.py / analyze_rto.py:
The original measurement scripts (failover_test*.py) label only the
POST-PROMOTION stabilisation interval as "rto_s". That is not total
recovery time: it excludes promotion_time_s, the interval from pod
deletion to a new primary being detected, which necessarily precedes
it. This script computes TOTAL RTO = promotion_time_s + rto_s, which
is the quantity reported as "RTO" throughout the paper (see
Section 3.5, "Live Multi-Engine Cluster Validation").

Required input files (place in the same directory as this script, or
pass --data-dir):
    rto_postgres_30tests.csv   (n=30, all valid)
    rto_mysql_clean.csv        (n=14, pre-filtered: excludes the
                                 test-15 timeout, rto_s=-1)
    rto_mongodb_30tests.csv    (n=30 raw; this script excludes the
                                 10 tests with an unresolved ~0.7s
                                 measurement artefact -- see below)

MongoDB exclusion, stated explicitly (do not silently change this
without updating Section 3.5 / 4.4 of the paper to match): 10 of the
30 MongoDB tests show a stabilisation time tightly clustered at
0.65-0.73s, implausibly fast and implausibly uniform next to the
remaining 20 tests (4.36-19.39s). We exclude these 10 as a likely
measurement artefact (most plausibly a stale replica-set status read
before Kubernetes fully evicted the terminating pod) rather than
treating them as genuine sub-second recoveries. The excluded test
numbers are listed in MONGODB_EXCLUDED_TESTS below for full
transparency and easy auditing.

Usage:
    python analyze_rto_total.py --data-dir .
"""

import argparse
import csv
import statistics
import json
from pathlib import Path

import numpy as np
from scipy import stats

# Test numbers excluded from the MongoDB analysis -- see module
# docstring. Change this list only alongside a corresponding update
# to the paper text (Section 3.5 / 4.4), not silently.
MONGODB_EXCLUDED_TESTS = {3, 4, 5, 13, 15, 16, 18, 19, 20, 21}

RANDOM_STATE = 42  # matches the seed used throughout the rest of the paper's code


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def total_rto(rows: list[dict]) -> np.ndarray:
    return np.array([
        float(r["promotion_time_s"]) + float(r["rto_s"]) for r in rows
    ])


def bca_ci(values: np.ndarray, n_resamples: int = 10_000):
    if len(values) < 2 or np.std(values) == 0:
        return float("nan"), float("nan")
    res = stats.bootstrap(
        (values,), np.mean, n_resamples=n_resamples,
        method="BCa", random_state=RANDOM_STATE,
    )
    return res.confidence_interval.low, res.confidence_interval.high


def summarize(name: str, values: np.ndarray) -> dict:
    lo, hi = bca_ci(values)
    return {
        "engine": name,
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "cv_pct": float(np.std(values, ddof=1) / np.mean(values) * 100),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "bca_ci_95": [float(lo), float(hi)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".",
                         help="directory containing the RTO CSV files")
    parser.add_argument("--out", default="results/rto_summary.json")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    pg_rows = load_csv(data_dir / "rto_postgres_30tests.csv")
    mysql_rows = load_csv(data_dir / "rto_mysql_clean.csv")
    mongo_rows_raw = load_csv(data_dir / "rto_mongodb_30tests.csv")
    mongo_rows = [r for r in mongo_rows_raw
                  if int(r["test"]) not in MONGODB_EXCLUDED_TESTS]

    print(f"MongoDB: {len(mongo_rows_raw)} raw tests, "
          f"{len(MONGODB_EXCLUDED_TESTS)} excluded "
          f"(tests {sorted(MONGODB_EXCLUDED_TESTS)}), "
          f"{len(mongo_rows)} used\n")

    engines = {
        "PostgreSQL": total_rto(pg_rows),
        "MySQL GR": total_rto(mysql_rows),
        "MongoDB RS": total_rto(mongo_rows),
    }

    summaries = {}
    print("=" * 65)
    print("TOTAL RTO BY ENGINE (promotion_time_s + rto_s)")
    print("=" * 65)
    for name, values in engines.items():
        s = summarize(name, values)
        summaries[name] = s
        print(f"\n{name} (n={s['n']})")
        print(f"  mean   : {s['mean']:.2f}s")
        print(f"  median : {s['median']:.2f}s")
        print(f"  std    : {s['std']:.2f}s")
        print(f"  CV     : {s['cv_pct']:.1f}%")
        print(f"  min/max: {s['min']:.2f}s / {s['max']:.2f}s")
        print(f"  BCa 95% CI: [{s['bca_ci_95'][0]:.2f}, {s['bca_ci_95'][1]:.2f}]")

    print("\n" + "=" * 65)
    print("MANN-WHITNEY U — pairwise, between engines")
    print("=" * 65)
    mw_results = []
    names = list(engines.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            u, p = stats.mannwhitneyu(engines[a], engines[b], alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{a} vs {b}: U={u:.1f}, p={p:.4g} {sig}")
            mw_results.append({"pair": f"{a} vs {b}", "U": float(u), "p": float(p)})

    # PostgreSQL temporal-ordering check (Section 4.4)
    pg_promo = [float(r["promotion_time_s"]) for r in pg_rows]
    first10 = statistics.mean(pg_promo[:10])
    rest = statistics.mean(pg_promo[10:])
    print("\n" + "=" * 65)
    print("POSTGRESQL — temporal ordering check on promotion_time_s")
    print("=" * 65)
    print(f"Tests 1-10 mean : {first10:.2f}s")
    print(f"Tests 11-30 mean: {rest:.2f}s")
    print("(A large gap here indicates promotion_time_s is not i.i.d. "
          "across test order -- see Section 4.4 / Threats to Validity.)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summaries": summaries,
            "mann_whitney": mw_results,
            "postgresql_temporal_check": {
                "tests_1_10_mean": first10,
                "tests_11_30_mean": rest,
            },
            "mongodb_excluded_tests": sorted(MONGODB_EXCLUDED_TESTS),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
