# signed_summary_v4.R
# parallel_sim_v3.R içindeki signed_summary'yi GEÇERSİZ KILAR (sonradan source edilir).
# Yeni ölçütler:
#  - obs_sign_agree : tüm gözlemler üzerinde işaret uyumu (v3 ile aynı)
#  - wagree         : |phi_or| ağırlıklı işaret uyumu (büyük atıflar daha çok sayılır)
#  - top50_agree    : her özellik için |phi_or| medyanının üstündeki gözlemlerde işaret uyumu
#  - sign_flip      : özellik düzeyinde ortalama işaret değişimi (v3; merkezli özelliklerde gürültülü)

signed_summary <- function(shap_fin, shap_or, n_signal = 7) {
  cn <- colnames(shap_or)
  shap_fin <- shap_fin[, cn, drop = FALSE]

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

  data.frame(
    feature        = cn,
    is_signal      = seq_along(cn) <= n_signal,
    mean_phi_fin   = mean_f,
    mean_phi_or    = mean_o,
    sign_flip      = sign(mean_f) != sign(mean_o),
    obs_sign_agree = obs_agree_all,
    wagree         = w_agree,
    top50_agree    = top50,
    share_fin      = imp_f / sum(imp_f),
    share_or       = imp_o / sum(imp_o),
    stringsAsFactors = FALSE
  )
}
