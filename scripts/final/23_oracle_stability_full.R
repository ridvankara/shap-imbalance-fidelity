# 23_oracle_stability_full.R - Oracle'in tohuma duyarlılığı, tüm 24 hücre.
# 22 ile aynı; 8 benzersiz oracle (2 korelasyon x 4 prevalans) x 6 tohum. Tahmini süre: ~30-45 dakika.
# 7777 tohumu mevcut oracle ile birebir eşleşmeli (DOGRULAMA satırı).
# Çıktı: results/oracle_stability/oracle_importance_full.csv, oracle_summary_full.csv

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")
source("R/oracle.R")

cell_meta      <- readRDS("data/cell_meta.rds")
eval_sets      <- readRDS("data/oracles/eval_sets_native.rds")
oracle_stored  <- readRDS("data/oracles/oracle_results_native.rds")

# Oracle yalnızca (korelasyon, prevalans) çiftine bağlı; n'ye bağlı değil.
# Her oracle_key için bir temsilci hücre seçiyoruz: 8 oracle.
okeys <- sapply(cell_meta, function(x) x$oracle_key)
keys  <- names(cell_meta)[!duplicated(okeys)]
cat("Oracle sayisi:", length(keys), "\n")
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
    rows[[length(rows) + 1]] <- data.frame(cell_key = key, oracle_key = meta$oracle_key, seed = sd,
                                           feature = names(imp), imp = as.numeric(imp),
                                           share = as.numeric(imp / sum(imp)),
                                           stringsAsFactors = FALSE)
    summ[[length(summ) + 1]] <- data.frame(cell_key = key, oracle_key = meta$oracle_key, seed = sd,
                                           auc = o$perf$auc, minutes = round(el, 1),
                                           total_imp = sum(imp), null_share = sum(imp[8:20]) / sum(imp))
    cat(sprintf("%s seed=%d AUC=%.4f minutes=%.1f\n", key, sd, o$perf$auc, el))
    write.csv(do.call(rbind, rows), "results/oracle_stability/oracle_importance_full.csv", row.names = FALSE)
    write.csv(do.call(rbind, summ), "results/oracle_stability/oracle_summary_full.csv", row.names = FALSE)
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
