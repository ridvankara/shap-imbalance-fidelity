# metrics.R - SHAP Karsilastirma Metrikleri

# Tum metrikler iki SHAP matrisi alir (oracle ve sonlu)
# Ortalama mutlak deger ile ozellik onem siralamasi cikarir

# Yardimci: SHAP matrisinden ozellik onem vektoru
shap_to_importance <- function(shap_matrix) {
  colMeans(abs(shap_matrix))
}

# 1. Kendall tau (tam siralama)
metric_kendall <- function(imp_finite, imp_oracle) {
  cor(imp_finite, imp_oracle, method = "kendall")
}

# 2. Spearman rho (tam siralama, monotonik)
metric_spearman <- function(imp_finite, imp_oracle) {
  cor(imp_finite, imp_oracle, method = "spearman")
}

# 3. Rank-biased overlap (top-k odakli, agirlikli)
metric_rbo <- function(imp_finite, imp_oracle, p = 0.9, k = NULL) {
  rank_f <- names(sort(imp_finite, decreasing = TRUE))
  rank_o <- names(sort(imp_oracle, decreasing = TRUE))
  if (is.null(k)) k <- length(rank_f)
  
  rbo_sum <- 0
  for (d in seq_len(k)) {
    set_f <- rank_f[seq_len(d)]
    set_o <- rank_o[seq_len(d)]
    intersection <- length(intersect(set_f, set_o))
    agreement <- intersection / d
    rbo_sum <- rbo_sum + (p^(d - 1)) * agreement
  }
  (1 - p) * rbo_sum
}

# 4. Jaccard top-k (set benzerligi)
metric_jaccard_topk <- function(imp_finite, imp_oracle, k) {
  top_f <- names(sort(imp_finite, decreasing = TRUE))[seq_len(k)]
  top_o <- names(sort(imp_oracle, decreasing = TRUE))[seq_len(k)]
  length(intersect(top_f, top_o)) / length(union(top_f, top_o))
}

# 5. Top-k recall (oraclein top-k uyelerinin sonlu top-kdeki orani)
metric_topk_recall <- function(imp_finite, imp_oracle, k) {
  top_f <- names(sort(imp_finite, decreasing = TRUE))[seq_len(k)]
  top_o <- names(sort(imp_oracle, decreasing = TRUE))[seq_len(k)]
  length(intersect(top_f, top_o)) / k
}

# 6. L1 distance on raw SHAP magnitudes (sirilamadan bagimsiz, atif buyuklugu)
metric_l1 <- function(imp_finite, imp_oracle) {
  # Hizalama icin ayni siralama adlari gerek
  ord <- names(imp_oracle)
  sum(abs(imp_finite[ord] - imp_oracle[ord]))
}

# 7. L2 distance
metric_l2 <- function(imp_finite, imp_oracle) {
  ord <- names(imp_oracle)
  sqrt(sum((imp_finite[ord] - imp_oracle[ord])^2))
}

# 8. Cosine similarity (yon benzerligi)
metric_cosine <- function(imp_finite, imp_oracle) {
  ord <- names(imp_oracle)
  v_f <- imp_finite[ord]
  v_o <- imp_oracle[ord]
  sum(v_f * v_o) / (sqrt(sum(v_f^2)) * sqrt(sum(v_o^2)))
}

# Hepsini bir kerede hesapla
compute_all_metrics <- function(shap_finite, shap_oracle, ks = c(5, 7, 10)) {
  imp_f <- shap_to_importance(shap_finite)
  imp_o <- shap_to_importance(shap_oracle)
  
  # Ortak ozellik adlari (guvenlik)
  common <- intersect(names(imp_f), names(imp_o))
  imp_f <- imp_f[common]
  imp_o <- imp_o[common]
  
  result <- list(
    kendall = metric_kendall(imp_f, imp_o),
    spearman = metric_spearman(imp_f, imp_o),
    rbo = metric_rbo(imp_f, imp_o, p = 0.9),
    l1 = metric_l1(imp_f, imp_o),
    l2 = metric_l2(imp_f, imp_o),
    cosine = metric_cosine(imp_f, imp_o)
  )
  
  for (k in ks) {
    result[[paste0("jaccard_k", k)]] <- metric_jaccard_topk(imp_f, imp_o, k)
    result[[paste0("recall_k", k)]] <- metric_topk_recall(imp_f, imp_o, k)
  }
  
  result
}

