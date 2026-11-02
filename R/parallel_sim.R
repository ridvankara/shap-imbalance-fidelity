# parallel_sim.R - Paralel simulasyon ve checkpoint mantigi

library(foreach)
library(doParallel)

run_one_rep <- function(cell_key, rep_id,
                         Sigma, beta_final, alpha_final, n_train,
                         oracle_shap, eval_set,
                         strategies, nrounds = 200,
                         base_seed = 1e6) {

  rep_seed <- base_seed + abs(digest::digest2int(paste(cell_key, rep_id)))

  train_dat <- generate_data(n_train, beta_final, alpha_final, Sigma,
                              seed = rep_seed)

  rep_results <- list()

  for (s in strategies) {
    res_one <- tryCatch({
      bal <- apply_balancing(train_dat$X, train_dat$y, s,
                              seed = rep_seed + 1)

      # ADAPTIF: n_effective = dengeleme sonrasi egitim seti boyutu
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

      data.frame(
        cell_key = cell_key,
        rep = rep_id,
        strategy = s,
        n_effective = n_eff,
        min_data_used = fit$min_data_used,
        auc = perf$auc,
        pr_auc = perf$pr_auc,
        brier = perf$brier,
        kendall = metrics$kendall,
        spearman = metrics$spearman,
        rbo = metrics$rbo,
        jaccard_k5 = metrics$jaccard_k5,
        jaccard_k7 = metrics$jaccard_k7,
        jaccard_k10 = metrics$jaccard_k10,
        recall_k5 = metrics$recall_k5,
        recall_k7 = metrics$recall_k7,
        l1 = metrics$l1,
        l2 = metrics$l2,
        cosine = metrics$cosine,
        error = NA_character_,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      data.frame(
        cell_key = cell_key,
        rep = rep_id,
        strategy = s,
        n_effective = NA,
        min_data_used = NA,
        auc = NA, pr_auc = NA, brier = NA,
        kendall = NA, spearman = NA, rbo = NA,
        jaccard_k5 = NA, jaccard_k7 = NA, jaccard_k10 = NA,
        recall_k5 = NA, recall_k7 = NA,
        l1 = NA, l2 = NA, cosine = NA,
        error = as.character(e$message),
        stringsAsFactors = FALSE
      )
    })

    rep_results[[s]] <- res_one
  }

  do.call(rbind, rep_results)
}

build_job_list <- function(cells, reps, checkpoint_dir) {
  jobs <- expand.grid(
    cell_key = cells,
    rep = reps,
    stringsAsFactors = FALSE
  )
  jobs$checkpoint_file <- sprintf("%s/cell_%s_rep_%03d.rds",
                                   checkpoint_dir, jobs$cell_key, jobs$rep)
  done <- file.exists(jobs$checkpoint_file)
  cat(sprintf("Toplam is: %d, tamamlanmis: %d, bekleyen: %d\n",
              nrow(jobs), sum(done), sum(!done)))
  jobs[!done, , drop = FALSE]
}

run_parallel_sim <- function(jobs, cell_meta, oracle_results, eval_sets,
                              strategies, checkpoint_dir,
                              n_cores = NULL, nrounds = 200) {

  if (is.null(n_cores)) n_cores <- max(1, parallel::detectCores() - 2)
  cat(sprintf("Paralel: %d core, %d is\n", n_cores, nrow(jobs)))

  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl), add = TRUE)

  t0 <- Sys.time()

  results <- foreach(
    i = seq_len(nrow(jobs)),
    .packages = c("lightgbm", "MASS", "themis", "pROC", "PRROC", "digest"),
    .export = c("run_one_rep", "generate_data", "apply_balancing",
                "fit_and_explain", "compute_all_metrics",
                "compute_performance", "default_hparams", "adaptive_min_data",
                "shap_to_importance", "metric_kendall", "metric_spearman",
                "metric_rbo", "metric_jaccard_topk", "metric_topk_recall",
                "metric_l1", "metric_l2", "metric_cosine"),
    .errorhandling = "pass"
  ) %dopar% {
    job <- jobs[i, ]
    meta <- cell_meta[[job$cell_key]]

    out <- run_one_rep(
      cell_key = job$cell_key,
      rep_id = job$rep,
      Sigma = meta$Sigma,
      beta_final = meta$beta_final,
      alpha_final = meta$alpha_final,
      n_train = meta$n_train,
      oracle_shap = oracle_results[[meta$oracle_key]]$shap,
      eval_set = eval_sets[[meta$oracle_key]],
      strategies = strategies,
      nrounds = nrounds
    )

    saveRDS(out, job$checkpoint_file)
    out
  }

  elapsed <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("\nTamamlandi: %.1f dakika\n", as.numeric(elapsed)))

  do.call(rbind, results)
}

collect_checkpoints <- function(checkpoint_dir) {
  files <- list.files(checkpoint_dir, pattern = "^cell_.*\\.rds$",
                       full.names = TRUE)
  if (length(files) == 0) return(NULL)
  cat(sprintf("Toplaniyor: %d dosya\n", length(files)))
  dfs <- lapply(files, readRDS)
  do.call(rbind, dfs)
}

