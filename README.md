# SHAP attribution fidelity under class-imbalance treatments

Code and results for the Monte Carlo study "Class-imbalance treatments and SHAP attribution fidelity: a Monte Carlo study of rank, magnitude and sign" (Rıdvan Kara, Hakkari University).

Six imbalance treatments (none, class weight, random over/under sampling, SMOTE, ADASYN) are compared on TreeSHAP attributions from LightGBM models. Finite-sample attributions are compared with an oracle trained on a large sample at native prevalence, over 24 design cells and 100 replications per cell (14,400 planned fits; 14,368 analysed).

## Layout

- `R/` simulation code: data-generating process (`dgp.R`), treatments (`balancing.R`), LightGBM + TreeSHAP (`modeling.R`), oracle (`oracle.R`), metrics (`metrics.R`), parallel runner (`parallel_sim.R`, `parallel_sim_v3.R`), and per-fit summaries (`signed_summary_v4.R`, `signed_summary_v5.R`; v5 supersedes v4 and adds total-importance columns).
- `scripts/final/` scripts used to produce the reported results, in order:
  - `06_build_cell_meta.R`: calibrated design per cell.
  - `09b_full_simulation_v2.R`: first full run (rank and magnitude metrics).
  - `18_run_v4_subset.R`, `19_run_v4_full.R`: 100-replication run with per-fit sign summaries.
  - `15_run_signed_v3.R`, `16_check_and_summary.R`: consistency checks against v2.
  - `12b_table1_v2.R`: strategy-rank table from the v2 results.
  - `17_bias_check.R`: check of the attribution intercept (read-only).
  - `20_run_v5_scale.R`: 100-replication run adding total-importance columns for scale-matching.
  - `21_iter_check.R`: null-share sensitivity to the number of boosting iterations (read-only).
- `scripts/analysis/` scripts that compute every number and figure in the manuscript from the result files:
  - `analysis.py` (fit-level dataset: scale-matched L1, Gini correlation), `tables.py` (main tables), `pareto.py` (Pareto frontiers, paired differences), `table1.py` (strategy ranks and within-cell correlations), `figs.py` (figures).
- `results/` result files used by the analysis:
  - `full_sim_v4_metrics.rds`, `full_sim_v4_signed.rds`: 100 replications.
  - `full_sim_v5_metrics.rds`, `full_sim_v5_signed.rds`: 100 replications with total-importance columns (metrics identical to v4).
  - `full_sim_v2_results.rds`: earlier run used for the first rank table.

## Definitions

- **Scale-matched L1:** the finite-sample importance vector is rescaled to have the same sum as the oracle vector, then the L1 distance is taken. Equivalently, it equals the oracle total times the L1 distance between importance shares.
- **Sign agreement:** observation-level agreement between finite-sample and oracle SHAP values for signal features. Null features give a chance baseline of about 0.50.
- **Gini correlation:** Yitzhaki's Gini correlation of finite-sample importances with the oracle ranking, averaged over the two directions.

## Reproducing

R 4.5, lightgbm 4.6, themis 1.0, pROC, PRROC, MASS, Matrix, digest, foreach, doParallel. Set the working directory to the repository root. The scripts use `setwd("C:/SHAP_imbalance")`; change this to the repository path.

**Not included in this repository:**
- `data/cell_meta.rds` and the oracle objects (`data/oracles/*.rds`). The oracle is deterministic given the seeds in `R/oracle.R`, so it can be rebuilt from `build_oracle_native()` and `build_eval_set_native()`. A driver script for this step is not yet in `scripts/final/`; the earlier driver (`04_build_oracles.R`) is out of date and has been removed.
- The Bayes AUCs for each cell (stored in `cell_meta.rds`).

## Known issues

- 28 SMOTE/ADASYN fits (π = 0.02, n = 500) fail with "Not enough observations of 1"; 4 random-under fits in the same cell return constant predictors. These are excluded from the analysis.
- `scripts/analysis/*.py` expect the result files in `results/` and write intermediate pickles to the current directory.

## Citation and licence

Cite the manuscript when using the code. Licence: [to be chosen by the author].

## Oracle stability (Section 3.7)

- `scripts/final/22_oracle_stability.R` and `23_oracle_stability_full.R`: refit each oracle with six training seeds (7777, 1001-1005). Note: oracles are identical across sample sizes for the same correlation and prevalence, so 23 refits duplicates (144 instead of 48); results are unaffected.
- `results/oracle_stability/`: importance and summary tables from script 23.
- `scripts/analysis/oracle_rescore/`: `rescore_all.py` rescored the finite-sample shares against each oracle (seed 7777 reproduces the stored Kendall tau to machine precision); `analyse.py` gives oracle-to-oracle agreement; `design.py` gives design-level means and paired differences (pairwise deletion).
