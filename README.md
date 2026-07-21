Cross-Database Generalisation of ML-Driven Replication Mode Selection
Code and data accompanying the paper "Testing the Preconditions for
Cross-Database Generalisation of Machine Learning-Driven Replication
Mode Selection: A Controlled Synthetic Study Across PostgreSQL,
MySQL, and MongoDB" (Jerrari & Assayad, submitted to the Journal of
Grid Computing, 2026).
Repository structure
```
.
├── data/
│   ├── classification/
│   │   ├── dataset_postgresql_644.csv   # n=644, shared generator, PostgreSQL params
│   │   ├── dataset_mysql_200.csv        # n=200, shared generator, MySQL params
│   │   └── dataset_mongodb_200.csv      # n=200, shared generator, MongoDB params
│   └── rto/
│       ├── rto_postgres_30tests.csv     # 30/30 successful failover tests
│       ├── rto_mysql_30tests.csv        # 15 attempted; test 15 timed out (rto_s=-1)
│       ├── rto_mysql_clean.csv          # 14 valid tests (test 15 excluded)
│       └── rto_mongodb_30tests.csv      # 30/30 returned a result; see Section 3.5
│                                          # of the paper for the 10 tests excluded
│                                          # from analysis (unresolved measurement
│                                          # artefact, not a failure)
├── k8s/
│   ├── postgres-cluster.yaml            # CloudNativePG v1.22.0, PostgreSQL 16.1
│   ├── mysql-cluster.yaml               # MySQL Operator, InnoDBCluster (Group Replication)
│   └── mongodb-replicaset.yaml          # bare StatefulSet, no dedicated operator
│                                          # (orchestration asymmetry, see Section 3.5)
├── scripts/
│   ├── common.py                        # shared config: seed, features, RF params
│   ├── generate_datasets.py             # the shared synthetic data generator
│   ├── train_classifiers.py             # RQ1: per-engine RF, repeated 10x5-fold CV
│   ├── stats_tests.py                   # BCa bootstrap, McNemar, Mann-Whitney
│   ├── sample_size_control.py           # RQ1b: PostgreSQL size-matched control
│   ├── compare_classifiers.py           # RQ1c: RF vs XGBoost/LogReg/SVM
│   ├── smote_availability.py            # RQ3: SMOTE diagnostic on AVAILABILITY
│   ├── analyze_rto_total.py             # RQ4: total RTO (promotion + stabilisation)
│   ├── failover_test.py                 # live failover harness — PostgreSQL
│   ├── failover_test_mongodb.py         # live failover harness — MongoDB
│   ├── failover_test_mysql.py           # live failover harness — MySQL (v1)
│   └── failover_test_mysql_v2.py        # live failover harness — MySQL (v2, adds
│                                          # automatic Group Replication recovery)
├── requirements.txt
├── LICENSE                              # MIT (code)
├── LICENSE-DATA                         # CC-BY-4.0 (datasets)
└── README.md
```
Reproducing the offline classification results
```bash
pip install -r requirements.txt
cd scripts

# Regenerate datasets from the shared generator (optional; CSVs are
# already included under data/classification/)
python generate_datasets.py

# Train per-engine classifiers, repeated stratified 5-fold CV (Table 3)
python train_classifiers.py --engine all --out-dir results

# Statistical tests: BCa bootstrap CIs, McNemar vs. heuristic, Mann-Whitney (Table 3, 4)
python stats_tests.py

# RQ1b: sample-size control, 20 stratified n=200 PostgreSQL subsamples (Table 5, 6)
python sample_size_control.py

# RQ1c: RandomForest vs. XGBoost, logistic regression, SVM (Table 8, 9)
python compare_classifiers.py --engine all

# RQ3: SMOTE diagnostic on PostgreSQL AVAILABILITY class (Section 4.5)
python smote_availability.py
```
All scripts read CSVs from the working directory by default; either
run them from inside `data/classification/` with the CSVs copied
alongside, or adjust `DATA_PATHS` in `common.py` to point at
`../data/classification/`.
Reproducing the live RTO validation
```bash
# Total RTO (promotion_time_s + stabilisation), Mann-Whitney, and the
# PostgreSQL temporal-ordering check (Section 4.4, Table 7)
cd scripts
python analyze_rto_total.py --data-dir ../data/rto
```
The `failover_test*.py` scripts are the harnesses that produced the
raw RTO CSVs against live Kubernetes clusters (see `k8s/` for the
corresponding cluster manifests). They require a running Kubernetes
context per engine (`kind-iaha-postgres`, `kind-iaha-mysql`,
`kind-iaha-mongodb`) and are included for methodological transparency
rather than one-shot reproducibility on an arbitrary machine — see
Section 3.5 of the paper for the exact protocol, sample sizes, and
known limitations (orchestration asymmetry, PostgreSQL temporal
ordering effect, MongoDB measurement artefact on 10/30 tests).
Data
All three classification datasets share an identical ten-feature
schema and are produced by the same synthetic generator
(`generate_datasets.py`), called with the same random seed and
varying only the requested sample size and an engine-specific
parameter. See Section 3.1 ("Datasets") and Section 3.2 ("Scope") of
the paper for full details on what this design does and does not
establish.
Dataset	n	CONSISTENCY	ECONOMIC	PERFORMANCE	AVAILABILITY
PostgreSQL	644	512 (79.5%)	57 (8.9%)	70 (10.9%)	5 (0.8%)
MySQL	200	157 (78.5%)	22 (11.0%)	18 (9.0%)	3 (1.5%)
MongoDB	200	170 (85.0%)	17 (8.5%)	12 (6.0%)	1 (0.5%)
PostgreSQL's larger sample size ($n=644$ vs. $n=200$) reflects reuse
of the dataset from our prior work (IAHA), not a design choice made
for this study; see Section 3.1 of the paper for why we did not
simply generate matching $n=644$ samples for MySQL and MongoDB.
Related work
This paper extends IAHA (Jerrari & Assayad, under review at the
Journal of Intelligent & Fuzzy Systems), which introduced the
replication-mode classifier for PostgreSQL that this study tests
across MySQL and MongoDB.
Citation
If you use this code or data, please cite:
```bibtex
@article{jerrari2026crossdb,
  title   = {Testing the Preconditions for Cross-Database Generalisation
             of Machine Learning-Driven Replication Mode Selection:
             A Controlled Synthetic Study Across PostgreSQL, MySQL,
             and MongoDB},
  author  = {Jerrari, Imane and Assayad, Ismail},
  journal = {Journal of Grid Computing},
  year    = {2026},
  note    = {Submitted}
}
```
License
Code: MIT License — see LICENSE
Data: CC-BY-4.0 — see LICENSE-DATA
Contact
Imane Jerrari — imane.jerrari-etu@etu.univh2c.ma
Laboratory of Information Systems (LIS), Hassan II University of Casablanca
