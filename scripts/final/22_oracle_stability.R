# 22_oracle_stability.R - Oracle'in tohuma duyarlılığı.
# 4 hücre x 6 tohum (7777 = mevcut oracle; birebir eşleşmeli). Yalnızca oracle yeniden eğitilir.
# Çıktı: results/oracle_stability/oracle_importance.csv, oracle_summary.csv

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")
source("R/oracle.R")

cell_meta      <- readRDS("data/cell_meta.rds")
eval_sets      <- readRDS("data/oracles/eval_sets_native.rds")
oracle_stored  <- readRDS("data/oracles/oracle_results_native.rds")

keys  <- c("independent_p02_n02000", "independent_p05_n02000",
           "independent_p10_n02000", "independent_p20_n02000")
seeds <- c(7777, 1001, 1002, 1003, 1004, 1005)

dir.create("results/oracle_stability", recursive = TRUE, showWarnings = FALSE)
rows <- list(); summ <- list()

for (key in keys) {
  meta <- cell_meta[[key]]
  prev <- as.numeric(sub(".*_p(\\d+)_.*", "\\1", key)) / 100
  ev   <- eval_sets[[meta$oracle_key]]
  for (sd in seeds) {
    t0 <- Sys.time()
    o  <- build_oracle_native(meta$beta_final, meta$alpha_final, meta$Sigma,
                              prevalence = prev, eval_set = ev,
                              seed = sd, nrounds = 500)
    el  <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    imp <- colMeans(abs(o$shap))
    rows[[length(rows) + 1]] <- data.frame(cell_key = key, seed = sd,
                                           feature = names(imp), imp = as.numeric(imp),
                                           share = as.numeric(imp / sum(imp)),
                                           stringsAsFactors = FALSE)
    summ[[length(summ) + 1]] <- data.frame(cell_key = key, seed = sd,
                                           auc = o$perf$auc, minutes = round(el, 1),
                                           total_imp = sum(imp), null_share = sum(imp[8:20]) / sum(imp))
    cat(sprintf("%s seed=%d AUC=%.4f minutes=%.1f\n", key, sd, o$perf$auc, el))
    write.csv(do.call(rbind, rows), "results/oracle_stability/oracle_importance.csv", row.names = FALSE)
    write.csv(do.call(rbind, summ), "results/oracle_stability/oracle_summary.csv", row.names = FALSE)
  }
  # Doğrulama: 7777 tohumu (döngüde zaten eğitildi) mevcut oracle ile aynı olmalı
  stored <- oracle_stored[[meta$oracle_key]]
  if (!is.null(stored)) {
    new_row <- rows[[length(rows) - length(seeds) + 1]]   # bu hücrenin 7777 satırı
    imp_s <- colMeans(abs(stored$shap))
    cat(sprintf("DOGRULAMA %s: max |fark| (7777 vs mevcut) = %.2e\n", key,
                max(abs(imp_s - new_row$imp))))
  }
}
cat("\nBitti.\n")
