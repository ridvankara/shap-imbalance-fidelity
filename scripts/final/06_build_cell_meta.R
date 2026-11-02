# 06_build_cell_meta.R - Hucre meta-verilerini olustur

setwd("C:/SHAP_imbalance")
source("R/dgp.R")

oracle_results <- readRDS("data/oracles/oracle_results.rds")

# Kalibrasyon tablosu
calib_table <- data.frame(
  corr_type = rep(c("independent", "block"), 4),
  prevalence = rep(c(0.20, 0.10, 0.05, 0.02), each = 2),
  scale = c(0.775, 0.439, 0.741, 0.420, 0.719, 0.406, 0.712, 0.402),
  alpha = c(-1.895, -1.895, -2.872, -2.875, -3.703, -3.699, -4.729, -4.714),
  stringsAsFactors = FALSE
)

n_train_values <- c(500, 2000, 10000)

cell_meta <- list()

for (i in seq_len(nrow(calib_table))) {
  corr <- calib_table$corr_type[i]
  prev <- calib_table$prevalence[i]
  scl <- calib_table$scale[i]
  alp <- calib_table$alpha[i]
  
  oracle_key <- sprintf("%s_p%02d", corr, round(prev * 100))
  Sigma <- build_sigma(20, corr)
  beta_raw <- build_beta(20, corr)
  beta_final <- scl * beta_raw
  
  for (n in n_train_values) {
    cell_key <- sprintf("%s_n%05d", oracle_key, n)
    cell_meta[[cell_key]] <- list(
      corr_type = corr,
      prevalence = prev,
      n_train = n,
      scale = scl,
      Sigma = Sigma,
      beta_final = beta_final,
      alpha_final = alp,
      oracle_key = oracle_key
    )
  }
}

saveRDS(cell_meta, "data/cell_meta.rds")
cat(sprintf("Hucre sayisi: %d\n", length(cell_meta)))
cat(sprintf("Hucre adlari ornek:\n"))
print(head(names(cell_meta), 10))

