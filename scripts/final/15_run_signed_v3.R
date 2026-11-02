# 15_run_signed_v3.R - v2 ile aynı tasarım; imzalı SHAP çıktılarıyla tam simülasyon.
# Checkpoint'ler checkpoints/full_sim_v3 altında; v2 dosyalarına dokunmaz.
# Tamamlanmış işler atlanır, eksik olanlar (ör. rep 77) otomatik çalışır.

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")
source("R/parallel_sim.R")
source("R/parallel_sim_v3.R")

library(digest)

cell_meta      <- readRDS("data/cell_meta.rds")
oracle_results <- readRDS("data/oracles/oracle_results_native.rds")
eval_sets      <- readRDS("data/oracles/eval_sets_native.rds")

strategies <- c("none", "random_over", "random_under",
                "smote", "adasyn", "class_weight")

cells <- names(cell_meta)
reps  <- 1:100

checkpoint_dir <- "checkpoints/full_sim_v3"
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

jobs <- build_job_list(cells, reps, checkpoint_dir)

cat(sprintf("\n=== FULL SIMULATION v3 (signed SHAP) ===\nStart: %s\n\n", Sys.time()))

if (nrow(jobs) > 0) {
  run_parallel_sim_v3(
    jobs = jobs,
    cell_meta = cell_meta,
    oracle_results = oracle_results,
    eval_sets = eval_sets,
    strategies = strategies,
    checkpoint_dir = checkpoint_dir,
    n_cores = 10
  )
}

all <- collect_v3(checkpoint_dir)
saveRDS(all$metrics, "results/full_sim_v3_metrics.rds")
saveRDS(all$signed,  "results/full_sim_v3_signed.rds")

cat(sprintf("\n=== DONE === %s\n", Sys.time()))
cat(sprintf("Metric satırı: %d (beklenen 14400)\n", nrow(all$metrics)))
cat(sprintf("Hata satırı: %d\n", sum(!is.na(all$metrics$error))))
