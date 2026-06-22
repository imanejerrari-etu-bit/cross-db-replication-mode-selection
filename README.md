# Cross-Database Replication Mode Selection — Data & Code

Supporting data and generation code for the paper:

> Jerrari, I., Assayad, I. "Cross-Database Generalisation of Machine
> Learning-Driven Replication Mode Selection: A Comparative Study
> Across PostgreSQL, MySQL, and MongoDB." *Journal of Grid Computing*
> (submitted).

This is a companion study to:

> Jerrari, I., Assayad, I. "IAHA: An Intelligent Adaptive High
> Availability Framework for Containerised PostgreSQL Databases Using
> Hybrid Machine Learning." *Journal of Intelligent & Fuzzy Systems*
> (under review).

## Contents

```
data/
  dataset_postgresql_644.csv   # n=644 samples, PostgreSQL
  dataset_mysql_200.csv        # n=200 samples, MySQL
  dataset_mongodb_200.csv      # n=200 samples, MongoDB
scripts/
  generate_datasets.py         # Single shared generator producing all three datasets
paper/
  CrossDB_JGridComputing_draft.tex   # Paper source (Springer Nature sn-jnl class)
  sn-jnl.cls                         # Springer Nature LaTeX class
  sn-mathphys.bst                    # Bibliography style
```

## Dataset description

Each dataset shares an identical ten-feature operational schema
(replication lag, transaction throughput, CPU utilisation, latency,
rollback rate, deadlocks, hour of day, memory usage, active
connections, and a generic write-ahead/journal buffer saturation
proxy), labelled with one of four replication modes: `CONSISTENCY`,
`ECONOMIC`, `PERFORMANCE`, `AVAILABILITY`.

All three datasets are produced by **a single shared Python
generator** (`scripts/generate_datasets.py`), called once per engine
with the same random seed (`np.random.seed(42)`) and the same
scenario-based sampling logic (five latent operating scenarios:
normal, peak, off-peak, crisis, degraded), varying only the requested
sample size and an engine-specific parameter that controls the
generation range of two features (`wal_buffers_full`,
`trans_rolled_back`). The four-class label is assigned by a
deterministic rule cascade over the sampled feature values, as
documented in the paper (Section 5.1, Limitation L1) and visible
directly in the script.

To regenerate the datasets:

```bash
pip install pandas numpy
python scripts/generate_datasets.py
```

## Citation

If you use this data or code, please cite the paper above. A formal
citation (with DOI) will be added once the paper is accepted.

## License

Code (`scripts/`) is licensed under the MIT License — see `LICENSE`.
Data (`data/`) is licensed under CC-BY-4.0 — see `data/LICENSE`.
