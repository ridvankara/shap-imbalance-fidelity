# 21_iter_check.R - Null payının iterasyon sayısına duyarlılığı (tek hücre, 3 replikasyon).
# Salt okunur; hiçbir dosya yazmaz. Oracle 500 iterasyonla, sonlu fit'ler 200 ile eğitiliyor;
# bu kontrol null payındaki farkın kısmen iterasyon farkından gelip gelmediğini gösterir.

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")
source("R/metrics.R")

library(digest)

cell_meta      <- readRDS("data/cell_meta.rds")
oracle_results <- readRDS("data/oracles/oracle_results_native.rds")
eval_sets      <- readRDS("data/oracles/eval_sets_native.rds")

key  <- "independent_p10_n02000"
meta <- cell_meta[[key]]
o    <- oracle_results[[meta$oracle_key]]
ev   <- eval_sets[[meta$oracle_key]]

# Null payı: 8.-20. özelliklerin mean|phi| toplamının oranı
null_share <- function(shap) {
  imp <- colMeans(abs(shap))
  sum(imp[8:20]) / sum(imp)
}

cat(sprintf("Oracle (500 iterasyon) null payi: %.3f\n", null_share(o$shap)))
cat("----------------------------------------------------------\n")

base_seed <- 1e6
for (rep_id in 1:3) {
  rep_seed <- base_seed + abs(digest::digest2int(paste(key, rep_id)))
  if (rep_seed > 2147480000) rep_seed <- rep_seed - 2147480000

  tr <- generate_data(meta$n_train, meta$beta_final, meta$alpha_final,
                      meta$Sigma, seed = rep_seed)

  for (s in c("none", "random_over")) {
    bal <- apply_balancing(tr$X, tr$y, s, seed = rep_seed + 1)
    for (nr in c(200, 500)) {
      fit <- fit_and_explain(bal$X, bal$y, bal$weights, ev$X,
                             nrounds = nr, seed = rep_seed + 2,
                             n_effective = nrow(bal$X))
      cat(sprintf("rep %d | %-12s | iterasyon %3d | null payi %.3f\n",
                  rep_id, s, nr, null_share(fit$shap)))
    }
  }
}
cat("----------------------------------------------------------\n")
cat("Yorum: iterasyon 500 ile null payi belirgin biçimde 0.087'ye yaklaşıyorsa,\n")
cat("oracle ile fark büyük ölçüde iterasyon sayısından kaynaklanıyor demektir.\n")
