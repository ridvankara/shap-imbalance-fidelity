# oracle.R - Adaptive Oracle (native eval set)

source("R/dgp.R")
source("R/modeling.R")
library(MASS)

oracle_n <- function(prevalence, target_positives = 30000) {
  ceiling(target_positives / prevalence)
}

# NATIVE eval set: native prevalence, generate_data ile dogrudan
build_eval_set_native <- function(beta, alpha, Sigma, n_eval = 5000, seed = 999) {
  set.seed(seed)
  generate_data(n_eval, beta, alpha, Sigma)
}

# Oracle egit + native eval setinde SHAP
build_oracle_native <- function(beta_final, alpha, Sigma, prevalence,
                                 eval_set, p = 20,
                                 target_positives = 30000,
                                 seed = 7777, nrounds = 500) {
  n_total <- oracle_n(prevalence, target_positives)
  cat(sprintf("  Oracle: prev=%.2f n=%d\n", prevalence, n_total))
  
  set.seed(seed)
  X_train <- mvrnorm(n_total, mu = rep(0, p), Sigma = Sigma)
  eta <- alpha + as.numeric(X_train %*% beta_final)
  p_y <- plogis(eta)
  y_train <- rbinom(n_total, 1, p_y)
  colnames(X_train) <- paste0("x", seq_len(p))
  
  cat(sprintf("  Pozitif: %d / Negatif: %d\n",
              sum(y_train == 1), sum(y_train == 0)))
  
  # Oracle: sabit hiperparametre (buyuk n, adaptif gerekmez)
  fit <- fit_and_explain(
    X_train = X_train, y_train = y_train,
    weights = rep(1, length(y_train)),
    X_eval = eval_set$X,
    nrounds = nrounds, seed = seed,
    n_effective = NULL  # oracle sabit
  )
  
  perf <- compute_performance(fit$preds, eval_set$y)
  imp <- shap_importance(fit$shap)
  
  cat(sprintf("  Oracle AUC (native eval) = %.3f\n", perf$auc))
  cat(sprintf("  Top 5: %s\n\n", paste(names(imp)[1:5], collapse = ", ")))
  
  list(shap = fit$shap, preds = fit$preds, perf = perf,
       importance = imp, n_train = n_total,
       Sigma = Sigma, beta = beta_final, alpha = alpha)
}

