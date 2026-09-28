# Cross-Database Generalisation of ML-Driven Replication Mode Selection

Code and data accompanying the paper **"Testing the Preconditions for
Cross-Database Generalisation of Machine Learning-Driven Replication
Mode Selection: A Controlled Synthetic Study Across PostgreSQL,
MySQL, and MongoDB"** (Jerrari & Assayad, submitted to the
*International Journal of Data Science and Analytics*, 2026;
minor revision).

This is a companion study to:
> Jerrari, I., Assayad, I. "IAHA: An Intelligent Adaptive High
> Availability Framework for Containerised PostgreSQL Databases Using
> Hybrid Machine Learning." *Journal of Intelligent & Fuzzy Systems*
> (under review).

## Repository structure

The repository is intentionally flat. This is the actual layout —
please open an issue if you find a discrepancy, rather than assume
the code has moved.

```
.
├── data/
│   ├── dataset_postgresql_644.csv     # n=644, shared generator, PostgreSQL params
│   ├── dataset_mysql_200.csv          # n=200, shared generator, MySQL params
│   ├── dataset_mongodb_200.csv        # n=200, shared generator, MongoDB params
│   └── LICENSE                        # CC-BY-4.0 (datasets)
├── rto_postgres_30tests.csv           # 30/30 successful failover tests
├── rto_mysql_30tests.csv              # 15 attempted; test 15 timed out
├── rto_mysql_clean.csv                # 14 valid tests (test 15 excluded)
├── rto_mysql_final.csv                # post-processed MySQL RTO series
├── rto_mongodb_30tests.csv            # 30/30 returned a result (10 flagged, see paper Sec. 3.7)
├── rto_mongodb_clean.csv              # 20 valid tests (artefact-affected 10 excluded)
├── rto_results.csv                    # combined/summary RTO output
├── postgres-cluster.yaml              # CloudNativePG v1.22.0, PostgreSQL 16.1
├── mysql-cluster.yaml                 # MySQL Operator, InnoDBCluster (Group Replication)
├── mongodb-cluster.yaml               # bare StatefulSet, no dedicated operator
│                                        # (orchestration asymmetry, see paper Sec. 3.7)
├── common.py                          # shared config: seed, features, RF params, data paths
├── generate_datasets.py               # the shared synthetic data generator
├── train_classifiers.py               # RQ1: per-engine RF, repeated 10x5-fold CV
├── per_class_metrics.py               # per-class precision/recall on pooled predictions
│                                        # (Table 5 of the paper; consistent with Table 3)
├── stats_tests.py                     # BCa bootstrap, McNemar, Mann-Whitney
├── sample_size_control.py             # RQ1b: PostgreSQL size-matched control
├── smote_availability.py              # RQ3: SMOTE diagnostic on AVAILABILITY
├── analyze_rto_total.py               # RQ4: total RTO (promotion + stabilisation)
├── compare_classifiers.py             # RQ1c: RF vs XGBoost/LogReg/SVM, identical CV protocol
├── failover_test_mongodb.py           # live failover harness — MongoDB
├── failover_test_mysql.py             # live failover harness — MySQL (v1)
├── failover_test_mysql_v2.py          # live failover harness — MySQL (v2, adds
│                                        # automatic Group Replication recovery)
├── failover_test_postgres.py          # live failover harness — PostgreSQL
├── mongodb_classifier_comparison.json     # RQ1c output — MongoDB
├── mysql_classifier_comparison.json       # RQ1c output — MySQL
├── postgresql_classifier_comparison.json  # RQ1c output — PostgreSQL
├── mongodb_per_class_metrics.json         # per-class precision/recall — MongoDB
├── mysql_per_class_metrics.json           # per-class precision/recall — MySQL
├── postgresql_per_class_metrics.json      # per-class precision/recall — PostgreSQL
├── paper/
│   ├── main.tex                       # paper source (Springer Nature sn-jnl class),
│   │                                    # matches the submitted revision exactly
│   ├── main.pdf                       # compiled PDF of the above
│   ├── sn-jnl.cls
│   └── sn-mathphys.bst
├── requirements.txt                   # minimum package versions
├── requirements-lock.txt              # exact pinned versions verified to reproduce
│                                        # Table 3 bit-for-bit (see "Reproducibility" below)
├── LICENSE                            # MIT (code)
└── README.md
```

All scripts referenced by the paper's pipeline (offline classification,
alternative-classifier comparison, and live failover for all three
engines) are now included in this repository.

## Reproducing the offline classification results

```bash
pip install -r requirements.txt
# or, for an exact, bit-for-bit reproduction of Table 3:
# pip install -r requirements-lock.txt

# Regenerate datasets from the shared generator (optional; CSVs are
# already included under data/)
python generate_datasets.py

# Train per-engine classifiers, repeated stratified 5-fold CV (Table 3)
python train_classifiers.py --engine all --out-dir results

# Statistical tests: BCa bootstrap CIs, McNemar vs. heuristic, Mann-Whitney (Table 3, 4)
python stats_tests.py

# RQ1b: sample-size control, 20 stratified n=200 PostgreSQL subsamples (Table 5, 6)
python sample_size_control.py

# RQ3: SMOTE diagnostic on PostgreSQL AVAILABILITY class (Section 4.6)
python smote_availability.py

# RQ1c: RandomForest vs. XGBoost, logistic regression, SVM (Table 7, 8)
python compare_classifiers.py --engine all --out-dir results

# Per-class precision/recall on pooled predictions (Table 5)
python per_class_metrics.py --engine all --out-dir results
```

All scripts are run from the repository root and read the CSVs from
`data/` via the paths set in `common.py` — no manual copying of files
required.

## Reproducing the live RTO validation

```bash
# Total RTO (promotion_time_s + stabilisation_s), Mann-Whitney, and the
# PostgreSQL temporal-ordering check (Section 4.7, Table 11)
python analyze_rto_total.py
```

The `failover_test_*.py` scripts (one per engine: PostgreSQL, MySQL,
MongoDB) are the harnesses that produced the raw RTO CSVs against
live Kubernetes clusters (see the `*-cluster.yaml` files for the
corresponding cluster manifests). They require a running Kubernetes
context per engine (`kind-iaha-postgres`, `kind-iaha-mysql`,
`kind-iaha-mongodb`) and are included for methodological transparency
rather than one-shot reproducibility on an arbitrary machine — see
Section 3.7 of the paper for the exact protocol, sample sizes, and
known limitations (orchestration asymmetry, PostgreSQL temporal
ordering effect, MongoDB measurement artefact on 10/30 tests).

## Reproducibility note on package versions

`requirements.txt` pins only minimum versions. We verified
independently that the exact versions in `requirements-lock.txt`
(numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1,
statsmodels 0.15.0, imbalanced-learn 0.14.2, under Python 3.13.14)
reproduce the mean accuracy and pooled macro-F1 in Table 3
**exactly**, on a clean install. If you reproduce with different
versions and get close-but-not-identical numbers, this is expected
(RandomForest's internals vary subtly across scikit-learn releases)
rather than a sign of a broken pipeline; see Section 3.7
("Experimental environment") of the paper.

## Data

All three classification datasets share an identical ten-feature
schema and are produced by the same synthetic generator
(`generate_datasets.py`), called with the same random seed and
varying only the requested sample size and an engine-specific
parameter. See Section 3.1 ("Datasets") and Section 3.2 ("Scope") of
the paper for full details on what this design does and does not
establish.

| Dataset    | n   | CONSISTENCY  | ECONOMIC   | PERFORMANCE | AVAILABILITY |
|------------|-----|--------------|------------|-------------|--------------|
| PostgreSQL | 644 | 512 (79.5%)  | 57 (8.9%)  | 70 (10.9%)  | 5 (0.8%)     |
| MySQL      | 200 | 157 (78.5%)  | 22 (11.0%) | 18 (9.0%)   | 3 (1.5%)     |
| MongoDB    | 200 | 170 (85.0%)  | 17 (8.5%)  | 12 (6.0%)   | 1 (0.5%)     |

PostgreSQL's larger sample size ($n=644$ vs. $n=200$) reflects reuse
of the dataset from our prior work (IAHA), not a design choice made
for this study; see Section 3.1 of the paper for why we did not
simply generate matching $n=644$ samples for MySQL and MongoDB.

## Citation

If you use this code or data, please cite:

```bibtex
@article{jerrari2026crossdb,
  title   = {Testing the Preconditions for Cross-Database Generalisation
             of Machine Learning-Driven Replication Mode Selection:
             A Controlled Synthetic Study Across PostgreSQL, MySQL,
             and MongoDB},
  author  = {Jerrari, Imane and Assayad, Ismail},
  journal = {International Journal of Data Science and Analytics},
  year    = {2026},
  note    = {Submitted; minor revision}
}
```

A formal citation (with DOI) will be added once the paper is accepted.

## License

- Code: MIT License — see `LICENSE`
- Data: CC-BY-4.0 — see `data/LICENSE`

## Contact

Imane Jerrari — imane.jerrari-etu@etu.univh2c.ma
Laboratory of Information Systems (LIS), Hassan II University of Casablanca
