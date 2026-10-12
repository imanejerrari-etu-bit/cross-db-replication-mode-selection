"""
feature_importance.py -- RQ2 (Section 4.4): cross-engine stability of
RandomForest feature importances.

Fits the per-engine RandomForest of common.make_classifier() on the full
dataset of each engine, prints the top-4 importances (Table 10 and Fig. 4)
and the Spearman rank correlation between the full ten-feature importance
rankings of each pair of engines (reported as rho = 0.806, 0.915 and
0.891; the smallest value is 0.806, so the paper states rho >= 0.80).

Usage:
    python feature_importance.py [--out-dir results]
"""

import argparse
import json
import os
import warnings

import numpy as np
from scipy.stats import spearmanr

from common import FEATURE_COLUMNS, LABEL_COLUMN, load_dataset, make_classifier

warnings.filterwarnings("ignore")
ENGINES = ["mongodb", "mysql", "postgresql"]


def importances(engine: str) -> np.ndarray:
    df = load_dataset(engine)
    clf = make_classifier()
    clf.fit(df[FEATURE_COLUMNS].to_numpy(), df[LABEL_COLUMN].to_numpy())
    return clf.feature_importances_


def main(out_dir: str) -> None:
    imp = {e: importances(e) for e in ENGINES}

    print("Top-4 feature importances (Table 10)")
    top = np.argsort(-imp["postgresql"])[:4]
    print(f"{'feature':22s}" + "".join(f"{e:>12s}" for e in ENGINES))
    for i in top:
        print(f"{FEATURE_COLUMNS[i]:22s}" + "".join(f"{imp[e][i]:12.3f}" for e in ENGINES))

    print("\nSpearman rank correlation between full importance rankings")
    rho = {}
    for a, b in [("mongodb", "mysql"), ("mongodb", "postgresql"), ("mysql", "postgresql")]:
        rho[f"{a}_vs_{b}"] = float(spearmanr(imp[a], imp[b])[0])
        print(f"  {a} vs {b}: rho = {rho[f'{a}_vs_{b}']:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
        json.dump({"features": FEATURE_COLUMNS,
                   "importances": {e: imp[e].tolist() for e in ENGINES},
                   "spearman": rho}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    main(ap.parse_args().out_dir)
