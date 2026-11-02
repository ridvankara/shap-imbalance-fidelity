# modeling.R - LightGBM + TreeSHAP

library(lightgbm)
library(pROC)
library(PRROC)

default_hparams <- function() {
  list(
    objective = "binary",
    metric = "binary_logloss",
    num_leaves = 31,
    learning_rate = 0.05,
    min_data_in_leaf = 20,
    feature_pre_filter = FALSE,
    verbose = -1,
    num_threads = 1
  )
}

adaptive_min_data <- function(n_effective) {
  max(5, floor(n_effective / 50))
}

fit_and_explain <- function(X_train, y_train, weights, X_eval,
                            hparams = NULL, nrounds = 200,
                            seed = NULL, n_effective = NULL) {
  if (is.null(hparams)) hparams <- default_hparams()
  if (!is.null(seed)) hparams$seed <- seed

  if (!is.null(n_effective)) {
    hparams$min_data_in_leaf <- adaptive_min_data(n_effective)
    hparams$min_data_in_bin <- 1
  }

  feature_names <- colnames(X_train)
  if (is.null(feature_names)) {
    feature_names <- paste0("x", seq_len(ncol(X_train)))
    colnames(X_train) <- feature_names
    colnames(X_eval) <- feature_names
  }

  dtrain <- lgb.Dataset(
    data = X_train,
    label = y_train,
    weight = weights,
    params = list(min_data_in_bin = if (!is.null(n_effective)) 1 else 3,
                  feature_pre_filter = FALSE)
  )

  model <- lgb.train(
    params = hparams,
    data = dtrain,
    nrounds = nrounds,
    verbose = -1
  )

  shap_full <- predict(model, X_eval, type = "contrib")
  shap <- shap_full[, -ncol(shap_full), drop = FALSE]
  colnames(shap) <- feature_names

  preds <- predict(model, X_eval)

  list(model = model,
       shap = shap,
       preds = preds,
       bias = shap_full[1, ncol(shap_full)],
       min_data_used = hparams$min_data_in_leaf)
}

compute_performance <- function(preds, y_true) {
  auc_roc <- as.numeric(pROC::roc(y_true, preds, quiet = TRUE)$auc)
  pr <- PRROC::pr.curve(scores.class0 = preds[y_true == 1],
                        scores.class1 = preds[y_true == 0],
                        curve = FALSE)
  brier <- mean((preds - y_true)^2)
  list(auc = auc_roc, pr_auc = pr$auc.integral, brier = brier)
}

shap_importance <- function(shap_matrix) {
  imp <- colMeans(abs(shap_matrix))
  sort(imp, decreasing = TRUE)
}

