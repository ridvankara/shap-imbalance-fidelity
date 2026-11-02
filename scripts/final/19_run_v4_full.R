# 19_run_v4_full.R - v4 tam koşusu (100 replikasyon).
# reps 1-10 zaten 18_run_v4_subset.R ile tamamlandı; build_job_list onları atlar, yalnızca kalan 2160 işi koşar.

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

reps <- 1:100
checkpoint_dir <- "checkpoints/full_sim_v4"
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

jobs <- build_job_list(cells, reps, checkpoint_dir)
cat(sprintf("\n=== v4 TAM KOSU === Baslangic: %s\n", Sys.time()))

if (nrow(jobs) > 0) {
  run_parallel_sim_v3(jobs = jobs, cell_meta = cell_meta,
                      oracle_results = oracle_results, eval_sets = eval_sets,
                      strategies = strategies, checkpoint_dir = checkpoint_dir,
                      n_cores = 10)
}

all <- collect_v3(checkpoint_dir)
saveRDS(all$metrics, "results/full_sim_v4_metrics.rds")
saveRDS(all$signed,  "results/full_sim_v4_signed.rds")
cat(sprintf("\n=== BITTI === %s\nMetrik satiri: %d (beklenen 14400)\nHata satiri: %d\n",
            Sys.time(), nrow(all$metrics), sum(!is.na(all$metrics$error))))
