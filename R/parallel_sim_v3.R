# parallel_sim_v3.R
# v2 ile AYNI eğitim çekilişleri ve aynı fit yolu; ek olarak imzalı SHAP çıktıları.
# Beklenen: metrikler (kendall, l1, auc vb.) v2 ile birebir aynı olmalı.
# Çıktı checkpoint'i: list(metrics = <data.frame>, signed = <data.frame>)

library(foreach)
library(doParallel)

# Özellik düzeyi imzalı özet. Eval seti hem finite hem oracle için aynı olduğundan
# gözlem bazında eşleştirme yapılabilir.
signed_summary <- function(shap_fin, shap_or, n_signal = 7) {
  cn <- colnames(shap_or)
  shap_fin <- shap_fin[, cn, drop = FALSE]

  mean_f <- colMeans(shap_fin)            # ortalama imzalı SHAP (özellik düzeyi)
  mean_o <- colMeans(shap_or)
  imp_f  <- colMeans(abs(shap_fin))       # mevcut magnitude tanımı
  imp_o  <- colMeans(abs(shap_or))

  data.frame(
    feature        = cn,
    is_signal      = seq_along(cn) <= n_signal,   # x1..x7 sinyal (beta != 0)
    mean_phi_fin   = mean_f,
    mean_phi_or    = mean_o,
    sign_flip      = sign(mean_f) != sign(mean_o),          # özellik düzeyi işaret değişimi
    obs_sign_agree = colMeans(sign(shap_fin) == sign(shap_or)),  # gözlem düzeyi işaret uyumu
    share_fin      = imp_f / sum(imp_f),  # ölçekten bağımsız paylar
    share_or       = imp_o / sum(imp_o),
    stringsAsFactors = FALSE
  )
}

run_one_rep_v3 <- function(cell_key, rep_id,
                           Sigma, beta_final, alpha_final, n_train,
                           oracle_shap, oracle_bias, eval_set,
                           strategies, nrounds = 200,
                           base_seed = 1e6) {

  # v2 ile birebir aynı seed mantığı
  rep_seed <- base_seed + abs(digest::digest2int(paste(cell_key, rep_id)))

  train_dat <- generate_data(n_train, beta_final, alpha_final, Sigma,
                             seed = rep_seed)

  metric_rows <- list()
  signed_rows <- list()

  for (s in strategies) {
    res <- tryCatch({
      bal <- apply_balancing(train_dat$X, train_dat$y, s,
                             seed = rep_seed + 1)
      n_eff <- nrow(bal$X)

      fit <- fit_and_explain(
        X_train = bal$X,
        y_train = bal$y,
        weights = bal$weights,
        X_eval = eval_set$X,
        nrounds = nrounds,
        seed = rep_seed + 2,
        n_effective = n_eff
      )

      metrics <- compute_all_metrics(fit$shap, oracle_shap)
      perf <- compute_performance(fit$preds, eval_set$y)
      sg <- signed_summary(fit$shap, oracle_shap)
      sg$cell_key <- cell_key
      sg$rep <- rep_id
      sg$strategy <- s
      sg$bias_fin <- fit$bias
      sg$bias_or <- oracle_bias

      list(
        metrics = data.frame(
          cell_key = cell_key, rep = rep_id, strategy = s,
          n_effective = n_eff, min_data_used = fit$min_data_used,
          auc = perf$auc, pr_auc = perf$pr_auc, brier = perf$brier,
          kendall = metrics$kendall, spearman = metrics$spearman,
          rbo = metrics$rbo,
          jaccard_k5 = metrics$jaccard_k5, jaccard_k7 = metrics$jaccard_k7,
          jaccard_k10 = metrics$jaccard_k10,
          recall_k5 = metrics$recall_k5, recall_k7 = metrics$recall_k7,
          l1 = metrics$l1, l2 = metrics$l2, cosine = metrics$cosine,
          error = NA_character_, stringsAsFactors = FALSE
        ),
        signed = sg
      )
    }, error = function(e) {
      list(
        metrics = data.frame(
          cell_key = cell_key, rep = rep_id, strategy = s,
          n_effective = NA, min_data_used = NA,
          auc = NA, pr_auc = NA, brier = NA,
          kendall = NA, spearman = NA, rbo = NA,
          jaccard_k5 = NA, jaccard_k7 = NA, jaccard_k10 = NA,
          recall_k5 = NA, recall_k7 = NA,
          l1 = NA, l2 = NA, cosine = NA,
          error = as.character(e$message), stringsAsFactors = FALSE
        ),
        signed = NULL
      )
    })
    metric_rows[[s]] <- res$metrics
    if (!is.null(res$signed)) signed_rows[[s]] <- res$signed
  }

  list(
    metrics = do.call(rbind, metric_rows),
    signed  = if (length(signed_rows) > 0) do.call(rbind, signed_rows) else NULL
  )
}

run_parallel_sim_v3 <- function(jobs, cell_meta, oracle_results, eval_sets,
                                strategies, checkpoint_dir,
                                n_cores = 10, nrounds = 200) {

  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl), add = TRUE)

  t0 <- Sys.time()

  foreach(
    i = seq_len(nrow(jobs)),
    .packages = c("lightgbm", "MASS", "themis", "pROC", "PRROC", "digest"),
    .export = c("run_one_rep_v3", "signed_summary", "generate_data",
                "apply_balancing", "fit_and_explain", "compute_all_metrics",
                "compute_performance", "default_hparams", "adaptive_min_data",
                "shap_to_importance", "metric_kendall", "metric_spearman",
                "metric_rbo", "metric_jaccard_topk", "metric_topk_recall",
                "metric_l1", "metric_l2", "metric_cosine"),
    .errorhandling = "pass"
  ) %dopar% {
    job  <- jobs[i, ]
    meta <- cell_meta[[job$cell_key]]
    o    <- oracle_results[[meta$oracle_key]]

    # Oracle'ın log-odds sabiti: logit(p) = bias + sum(phi)
    oracle_bias <- mean(qlogis(o$preds) - rowSums(o$shap))

    out <- run_one_rep_v3(
      cell_key = job$cell_key,
      rep_id = job$rep,
      Sigma = meta$Sigma,
      beta_final = meta$beta_final,
      alpha_final = meta$alpha_final,
      n_train = meta$n_train,
      oracle_shap = o$shap,
      oracle_bias = oracle_bias,
      eval_set = eval_sets[[meta$oracle_key]],
      strategies = strategies,
      nrounds = nrounds
    )

    saveRDS(out, job$checkpoint_file)
    out
  }

  cat(sprintf("\nTamamlandi: %.1f dakika\n",
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

collect_v3 <- function(checkpoint_dir) {
  files <- list.files(checkpoint_dir, pattern = "^cell_.*\\.rds$",
                      full.names = TRUE)
  cat(sprintf("Toplaniyor: %d dosya\n", length(files)))
  objs <- lapply(files, readRDS)
  list(
    metrics = do.call(rbind, lapply(objs, function(o) o$metrics)),
    signed  = do.call(rbind, lapply(objs, function(o) o$signed))
  )
}
