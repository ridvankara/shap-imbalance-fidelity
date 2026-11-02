# 16_check_and_summary.R
# (1) v3 metrikleri v2 ile birebir eşleşiyor mu? (aynı seed, aynı fit yolu)
# (2) Pareto için ortalamalar, (3) sign-flip ve intercept/ölçek analizi.

setwd("C:/SHAP_imbalance")
library(dplyr)

v2 <- readRDS("results/full_sim_v2_results.rds")
v3 <- readRDS("results/full_sim_v3_metrics.rds")
sg <- readRDS("results/full_sim_v3_signed.rds")

# ---- (1) Birebir kontrol
key <- c("cell_key", "rep", "strategy")
j <- inner_join(v2, v3, by = key, suffix = c("_v2", "_v3"))
cat(sprintf("Eşleşen satır: %d\n", nrow(j)))
for (m in c("kendall", "l1", "l2", "auc")) {
  d <- max(abs(j[[paste0(m, "_v2")]] - j[[paste0(m, "_v3")]]), na.rm = TRUE)
  cat(sprintf("  %-8s maks fark: %.3g\n", m, d))
}
cat("(Beklenen: 0 veya makine hassasiyetinde; değilse seed/fit yolu farklı demektir.)\n\n")

# ---- (2) Ortalamalar ve Pareto
means <- v3 %>%
  filter(!is.na(kendall)) %>%
  group_by(strategy) %>%
  summarise(tau = mean(kendall), L1 = mean(l1), n_fits = n(), .groups = "drop")
cat("=== Ortalamalar (v3) ===\n"); print(as.data.frame(means))

# Baskınlık: B, A'yı baskılar ancak tau_B >= tau_A ve L1_B <= L1_A (biri katı)
dom <- function(a, b) with(means, tau[strategy == b] >= tau[strategy == a] &
                                  L1[strategy == b] <= L1[strategy == a] &
                                  (tau[strategy == b] > tau[strategy == a] |
                                   L1[strategy == b] < L1[strategy == a]))
cat("\n=== Hangi strateji kimi baskılıyor ===\n")
for (a in means$strategy) for (b in means$strategy) if (a != b && dom(a, b))
  cat(sprintf("  %s baskılanıyor: %s tarafından\n", a, b))

# ---- (3) İşaret ve ölçek analizi
sg <- sg %>% mutate(
  prevalence = as.numeric(sub(".*_p(\\d+)_.*", "\\1", cell_key)) / 100,
  n_train    = as.integer(sub(".*_n(\\d+)$", "\\1", cell_key))
)

cat("\n=== Özellik düzeyi işaret değişimi oranı (sinyal vs gürültü) ===\n")
flip_tab <- sg %>%
  group_by(strategy, is_signal) %>%
  summarise(flip_rate = mean(sign_flip), obs_sign_agree = mean(obs_sign_agree),
            .groups = "drop")
print(as.data.frame(flip_tab))

cat("\n=== Prevalans x strateji: sinyal özelliklerde işaret değişimi ===\n")
print(as.data.frame(
  sg %>% filter(is_signal) %>%
    group_by(strategy, prevalence) %>%
    summarise(flip_rate = mean(sign_flip), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = prevalence, values_from = flip_rate)
))

# Magnitude farkının ölçek kaymasından mı, şekil farkından mı geldiğini ayırmak için:
# paylaşım-tabanlı L1 (ölçekten bağımsız) ve intercept farkı.
cat("\n=== Ölçeğe duyarsız L1 (paylar) ve logit intercept farkı ===\n")
fit_level <- sg %>%
  group_by(cell_key, rep, strategy) %>%
  summarise(share_l1 = sum(abs(share_fin - share_or)),
            bias_diff = first(bias_fin - bias_or), .groups = "drop")
print(as.data.frame(
  fit_level %>% group_by(strategy) %>%
    summarise(share_l1 = mean(share_l1, na.rm = TRUE),
              bias_diff = mean(bias_diff, na.rm = TRUE), .groups = "drop")
))

# Oracle eşleşmesi: her fit'in oracle'la aynı eval setinde karşılaştırıldığı göz önüne alınarak
# bias_diff'in ortalaması, her strateji için logit ölçek kaymasının büyüklüğünü verir.
cat("\nTamam.\n")
