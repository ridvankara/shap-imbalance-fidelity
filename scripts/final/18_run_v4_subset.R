# 18_run_v4_subset.R - v4 ölçütleri (wagree, top50_agree) ile çalıştırma.
# Metrikler v3 ile aynı tohumları kullandığı için reps 1-10 metrikleri v3 ile birebir aynı çıkmalı.
# Önce alt küme (reps 1:10, ~240 iş); tam koşu için reps <- 1:100.

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")
source("R/parallel_sim.R")
source("R/parallel_sim_v3.R")   # run_one_rep_v3, run_parallel_sim_v3, collect_v3
source("R/signed_summary_v4.R") # signed_summary'yi geçersiz kılar

library(digest)

cell_meta      <- readRDS("data/cell_meta.rds")
oracle_results <- readRDS("data/oracles/oracle_results_native.rds")
eval_sets      <- readRDS("data/oracles/eval_sets_native.rds")

strategies <- c("none", "random_over", "random_under", "smote", "adasyn", "class_weight")
cells      <- names(cell_meta)

reps <- 1:10                     # <- tam koşu için 1:100 yapın
checkpoint_dir <- "checkpoints/full_sim_v4"
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

jobs <- build_job_list(cells, reps, checkpoint_dir)
if (nrow(jobs) > 0) {
  run_parallel_sim_v3(jobs = jobs, cell_meta = cell_meta,
                      oracle_results = oracle_results, eval_sets = eval_sets,
                      strategies = strategies, checkpoint_dir = checkpoint_dir,
                      n_cores = 10)
}

all <- collect_v3(checkpoint_dir)
saveRDS(all$metrics, "results/full_sim_v4_metrics.rds")
saveRDS(all$signed,  "results/full_sim_v4_signed.rds")
cat(sprintf("\nBitti. Metrik satiri: %d, imzali satir: %d\n", nrow(all$metrics), nrow(all$signed)))
