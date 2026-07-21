"""
common.py — shared configuration for all analysis scripts.

Centralising these constants matters for reproducibility: every
script (train_classifiers.py, stats_tests.py, sample_size_control.py,
smote_availability.py) imports RANDOM_STATE and the RandomForest
hyperparameters from here, so a reviewer re-running the pipeline gets
the same folds and the same trees every time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------
# Fixed seed for every stochastic step in the pipeline (RF training,
# fold assignment in RepeatedStratifiedKFold, bootstrap resampling,
# stratified subsampling in sample_size_control.py, and SMOTE).
# Matches the seed already used in scripts/generate_datasets.py.
# ---------------------------------------------------------------
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "replication_lag_mb",
    "transactions_per_sec",
    "cpu_percent",
    "latency_ms",
    "trans_rolled_back",
    "deadlocks",
    "hour_of_day",
    "memory_usage_pct",
    "connections_active",
    "wal_buffers_full",
]

LABEL_COLUMN = "mode"  # adjust if your CSV uses a different column name
CLASSES = ["CONSISTENCY", "ECONOMIC", "PERFORMANCE", "AVAILABILITY"]

DATA_PATHS = {
    "postgresql": "dataset_postgresql_644.csv",
    "mysql": "dataset_mysql_200.csv",
    "mongodb": "dataset_mongodb_200.csv",
}


def load_dataset(engine: str) -> pd.DataFrame:
    """Load one engine's CSV. Raises a clear error if the expected
    columns are not found, rather than failing silently downstream."""
    path = DATA_PATHS[engine]
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing expected column(s) {missing}. "
            f"If your CSV uses different column names, update "
            f"FEATURE_COLUMNS / LABEL_COLUMN in common.py to match."
        )
    return df


def make_classifier() -> RandomForestClassifier:
    """The single RandomForest configuration used everywhere in the
    paper: 200 trees, balanced class weighting (Section 3.2,
    Classification Protocol)."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def cpu_heuristic(df: pd.DataFrame) -> np.ndarray:
    """Static CPU-threshold heuristic used as the baseline in
    Table 3 / Table 4 (Section 3.3, Statistical Testing):
      cpu_percent < 30           -> ECONOMIC
      30 <= cpu_percent < 50     -> PERFORMANCE
      otherwise                  -> CONSISTENCY
    The heuristic cannot express AVAILABILITY by construction."""
    cpu = df["cpu_percent"].to_numpy()
    pred = np.full(len(df), "CONSISTENCY", dtype=object)
    pred[cpu < 30] = "ECONOMIC"
    pred[(cpu >= 30) & (cpu < 50)] = "PERFORMANCE"
    return pred
