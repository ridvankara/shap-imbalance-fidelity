# 09b_full_simulation_v2.R - Native eval + adaptif hiperparametre

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")
source("R/parallel_sim.R")

library(digest)

cell_meta <- readRDS("data/cell_meta.rds")
oracle_results <- readRDS("data/oracles/oracle_results_native.rds")
eval_sets <- readRDS("data/oracles/eval_sets_native.rds")

strategies <- c("none", "random_over", "random_under",
                "smote", "adasyn", "class_weight")

cells <- names(cell_meta)
reps <- 1:100

checkpoint_dir <- "checkpoints/full_sim_v2"
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

jobs <- build_job_list(cells, reps, checkpoint_dir)

cat(sprintf("\n=== FULL SIMULATION v2 (native eval + adaptive params) ===\n"))
cat(sprintf("Start: %s\n\n", Sys.time()))

if (nrow(jobs) > 0) {
  results <- run_parallel_sim(
    jobs = jobs,
    cell_meta = cell_meta,
    oracle_results = oracle_results,
    eval_sets = eval_sets,
    strategies = strategies,
    checkpoint_dir = checkpoint_dir,
    n_cores = 10
  )
}

all_results <- collect_checkpoints(checkpoint_dir)
saveRDS(all_results, "results/full_sim_v2_results.rds")

cat(sprintf("\n=== DONE === %s\n", Sys.time()))
cat(sprintf("Total rows: %d\n", nrow(all_results)))

n_errors <- sum(!is.na(all_results$error))
if (n_errors > 0) {
  cat(sprintf("Errors: %d\n", n_errors))
  print(table(all_results$cell_key[!is.na(all_results$error)],
              all_results$strategy[!is.na(all_results$error)]))
}

