# signed_summary_v5.R - v4 ölçütleri + toplam mutlak önem (ölçek ayrıştırması için).
# total_fin/total_or: sum_j mean|phi_j| (ölçek); null_fin/null_or: null özelliklerin toplam payı.

signed_summary <- function(shap_fin, shap_or, n_signal = 7) {
  cn <- colnames(shap_or)
  shap_fin <- shap_fin[, cn, drop = FALSE]
  sig <- seq_along(cn) <= n_signal

  mean_f <- colMeans(shap_fin)
  mean_o <- colMeans(shap_or)
  imp_f  <- colMeans(abs(shap_fin))
  imp_o  <- colMeans(abs(shap_or))

  agree <- sign(shap_fin) == sign(shap_or)
  ao    <- abs(shap_or)

  obs_agree_all <- colMeans(agree)
  w_agree       <- colSums(agree * ao) / colSums(ao)
  top50 <- vapply(seq_along(cn), function(j) {
    idx <- ao[, j] >= median(ao[, j])
    mean(agree[idx, j])
  }, numeric(1))

  tot_f <- sum(imp_f); tot_o <- sum(imp_o)

  data.frame(
    feature        = cn,
    is_signal      = sig,
    mean_phi_fin   = mean_f,
    mean_phi_or    = mean_o,
    sign_flip      = sign(mean_f) != sign(mean_o),
    obs_sign_agree = obs_agree_all,
    wagree         = w_agree,
    top50_agree    = top50,
    share_fin      = imp_f / tot_f,
    share_or       = imp_o / tot_o,
    total_fin      = tot_f,
    total_or       = tot_o,
    null_fin       = sum(imp_f[!sig]),
    null_or        = sum(imp_o[!sig]),
    stringsAsFactors = FALSE
  )
}
