# 17_bias_check.R - Taban değer (bias) doğrulaması. Salt okunur; hiçbir şey yazmaz.
# Amaç: fit$bias (predict type="contrib" son sütunu) ile eğitim verisindeki ham skorun
# ortalaması tutarlı mı, ve logit(prevalans) ile ne kadar uyumlu?

setwd("C:/SHAP_imbalance")
source("R/dgp.R")
source("R/balancing.R")
source("R/modeling.R")

cell_meta <- readRDS("data/cell_meta.rds")
eval_sets <- readRDS("data/oracles/eval_sets_native.rds")

cells_to_check <- c("independent_p02_n00500", "independent_p02_n02000", "independent_p20_n02000")
strategies     <- c("none", "random_over")

for (key in cells_to_check) {
  meta <- cell_meta[[key]]
  tr   <- generate_data(meta$n_train, meta$beta_final, meta$alpha_final,
                        meta$Sigma, seed = 123)
  cat(sprintf("\n== %s  (train prevalans = %.4f, logit = %.3f)\n",
              key, mean(tr$y), qlogis(mean(tr$y))))

  for (s in strategies) {
    bal <- apply_balancing(tr$X, tr$y, s, seed = 124)
    fit <- fit_and_explain(bal$X, bal$y, bal$weights, eval_sets[[meta$oracle_key]]$X,
                           nrounds = 200, seed = 125, n_effective = nrow(bal$X))

    raw_train <- predict(fit$model, bal$X, type = "raw")
    raw_eval  <- predict(fit$model, eval_sets[[meta$oracle_key]]$X, type = "raw")
    contrib_sum_eval <- rowSums(fit$shap)

    cat(sprintf(
      "  %-12s bias(contrib)=%7.3f | mean raw(train)=%7.3f | mean(raw_eval - sum phi)=%7.3f | logit(mean y_bal)=%6.3f\n",
      s, fit$bias, mean(raw_train), mean(raw_eval - contrib_sum_eval),
      qlogis(mean(bal$y))))
  }
}
cat("\nYorum: bias(contrib) ile mean(raw_eval - sum phi) aynı çıkmalı (TreeSHAP toplamsallık).\n")
cat("mean(raw_train) ile bias(contrib) farklıysa bias çıkarımında sorun var demektir.\n")
